from __future__ import annotations

import copy
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from cloud.training.parameter_freeze import unwrap_trainable_module
from edge.sample_quality import (
    HIGH_QUALITY,
    LOW_QUALITY,
    EntropyQualityClassifier,
    EntropyQualityStats,
)
from edge.window_drift_detector import WindowDriftDetector
from model_management.model_zoo import build_detection_model, get_model_family

_SIGMOID_FOREGROUND_PROBABILITY_FLOOR = 0.5


@dataclass(frozen=True)
class _BufferedSample:
    frame_id: int
    frame: np.ndarray
    artifacts: dict[str, Any]
    latency_ms: float | None


@dataclass(frozen=True)
class PendingLocalTTAUpdate:
    trigger_frame_id: int
    trained_state_dict: dict[str, Any]
    model_version_before: str
    snapshot_lock_ms: float
    batch_size: int
    num_epoch: int
    initial_loss: float
    loss: float
    initial_selected_logit_count: int
    selected_logit_count: int
    initial_gate_stats: dict[str, Any]
    gate_stats: dict[str, Any]
    trainable_param_count: int
    low_quality_sample_count: int
    started_perf: float
    shadow_train_ms: float
    live_training_mode: bool
    live_module_training_state: dict[str, bool]


class _TTASkip(RuntimeError):
    def __init__(self, reason: str, **details: Any) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = dict(details)


class TTADetectionAdapter:
    """Small differentiable-output adapter for local detector TTA."""

    def __init__(
        self,
        detector: object,
        *,
        model_override: object | None = None,
        entropy_margin_ratio: float = 0.4,
        adaptive_entropy_gate: bool = False,
        max_entropy_margin_ratio: float = 0.7,
        min_selected_logit_count: int = 16,
    ) -> None:
        self.detector = detector
        self.model = (
            model_override if model_override is not None else getattr(detector, "model", detector)
        )
        self.trainable_model = unwrap_trainable_module(
            self.model,
            model_name=str(getattr(detector, "model_name", "")),
        )
        self.entropy_margin_ratio = max(0.0, min(1.0, float(entropy_margin_ratio)))
        self.adaptive_entropy_gate = bool(adaptive_entropy_gate)
        self.max_entropy_margin_ratio = max(
            self.entropy_margin_ratio,
            min(1.0, max(0.0, float(max_entropy_margin_ratio))),
        )
        self.min_selected_logit_count = max(1, int(min_selected_logit_count))

    def build_batch(self, frames: list[np.ndarray]) -> list[torch.Tensor]:
        if not frames:
            raise _TTASkip("empty_batch")
        prepare = getattr(self.detector, "_prepare_image_tensor", None)
        if not callable(prepare):
            raise _TTASkip("preprocess_unavailable")
        return [prepare(frame) for frame in frames]

    def forward_tta_outputs(
        self,
        batch: list[torch.Tensor],
        *,
        augment: bool = False,
    ) -> Any:
        images = [torch.flip(image, dims=(-1,)) for image in batch] if augment else batch
        custom_forward = getattr(self.model, "forward_tta_outputs", None)
        if callable(custom_forward):
            return custom_forward(images, augment=augment)

        model = self.model
        rfdetr_outputs = self._forward_rfdetr(model, images)
        if rfdetr_outputs is not None:
            return rfdetr_outputs

        model_type = type(model).__name__
        if model_type in {"YOLODetectionModel", "RTDETRDetectionModel"}:
            return self._forward_ultralytics(model, images)

        raise _TTASkip("logits_unavailable")

    def entropy_loss(self, outputs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        logits, mode = _extract_differentiable_logits(self.model, outputs)
        if logits is None or not isinstance(logits, torch.Tensor) or not logits.requires_grad:
            raise _TTASkip("logits_unavailable")
        rows = _logit_rows(logits)
        if rows is None or rows.numel() == 0 or rows.shape[-1] <= 1:
            raise _TTASkip("logits_unavailable")
        if _background_is_last(mode) and rows.shape[-1] > 1:
            rows = rows[:, :-1]
        if rows.shape[-1] <= 1:
            raise _TTASkip("logits_unavailable")

        if str(mode).startswith("sigmoid"):
            probs = torch.sigmoid(rows)
            p = probs.max(dim=-1).values.clamp(1.0e-8, 1.0 - 1.0e-8)
            foreground_mask = p.detach() >= _SIGMOID_FOREGROUND_PROBABILITY_FLOOR
            if not bool(foreground_mask.any()):
                raise _TTASkip("no_foreground_logits")
            entropy = -((p * torch.log(p)) + ((1.0 - p) * torch.log(1.0 - p)))
            entropy = entropy / math.log(2.0)
            entropy = entropy[foreground_mask]
        else:
            probs = torch.softmax(rows, dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1.0e-8))).sum(dim=-1)
            entropy = entropy / max(math.log(max(2, int(rows.shape[-1]))), 1.0e-8)

        selected = entropy
        strict_selected_count = int(entropy.numel())
        max_entropy_candidate_count = int(entropy.numel())
        adaptive_entropy_gate_used = False
        if self.entropy_margin_ratio > 0.0:
            strict_mask = entropy <= self.entropy_margin_ratio
            strict_selected_count = int(strict_mask.sum().item())
            max_entropy_mask = entropy <= self.max_entropy_margin_ratio
            max_entropy_candidate_count = int(max_entropy_mask.sum().item())
            if strict_selected_count >= self.min_selected_logit_count:
                selected = entropy[strict_mask]
            elif self.adaptive_entropy_gate:
                if max_entropy_candidate_count < self.min_selected_logit_count:
                    raise _TTASkip(
                        "insufficient_reliable_logits",
                        required_selected_logit_count=self.min_selected_logit_count,
                        actual_selected_logit_count=max_entropy_candidate_count,
                        strict_selected_logit_count=strict_selected_count,
                        max_entropy_candidate_count=max_entropy_candidate_count,
                        entropy_margin_ratio=self.entropy_margin_ratio,
                        max_entropy_margin_ratio=self.max_entropy_margin_ratio,
                        adaptive_entropy_gate=True,
                    )
                eligible = entropy[max_entropy_mask]
                selected = torch.topk(
                    eligible,
                    k=self.min_selected_logit_count,
                    largest=False,
                    sorted=True,
                ).values
                adaptive_entropy_gate_used = True
            elif strict_selected_count > 0:
                selected = entropy[strict_mask]
            else:
                raise _TTASkip(
                    "no_reliable_logits",
                    required_selected_logit_count=self.min_selected_logit_count,
                    actual_selected_logit_count=0,
                    strict_selected_logit_count=0,
                    max_entropy_candidate_count=max_entropy_candidate_count,
                    entropy_margin_ratio=self.entropy_margin_ratio,
                    max_entropy_margin_ratio=self.max_entropy_margin_ratio,
                    adaptive_entropy_gate=False,
                )
        loss = selected.mean()
        return loss, {
            "logit_count": int(rows.shape[0]),
            "foreground_logit_count": int(entropy.numel()),
            "strict_selected_logit_count": strict_selected_count,
            "max_entropy_candidate_count": max_entropy_candidate_count,
            "selected_logit_count": int(selected.numel()),
            "entropy": float(entropy.detach().mean().item()),
            "effective_entropy_threshold": float(selected.detach().max().item()),
            "entropy_margin_ratio": self.entropy_margin_ratio,
            "max_entropy_margin_ratio": self.max_entropy_margin_ratio,
            "adaptive_entropy_gate": self.adaptive_entropy_gate,
            "adaptive_entropy_gate_used": adaptive_entropy_gate_used,
            "required_selected_logit_count": self.min_selected_logit_count,
        }

    def consistency_loss(self, outputs_a: Any, outputs_b: Any) -> torch.Tensor | None:
        logits_a, mode_a = _extract_differentiable_logits(self.model, outputs_a)
        logits_b, mode_b = _extract_differentiable_logits(self.model, outputs_b)
        if (
            not isinstance(logits_a, torch.Tensor)
            or not isinstance(logits_b, torch.Tensor)
            or tuple(logits_a.shape) != tuple(logits_b.shape)
            or mode_a != mode_b
        ):
            return None
        aligned_b = _align_horizontally_flipped_logits(logits_b, outputs_b)
        if aligned_b is None:
            # Flattened dense-detector anchors and DETR queries cannot be
            # aligned safely without their spatial layout or a matcher.
            return None
        rows_a = _logit_rows(logits_a)
        rows_b = _logit_rows(aligned_b)
        if rows_a is None or rows_b is None or tuple(rows_a.shape) != tuple(rows_b.shape):
            return None
        if _background_is_last(mode_a) and rows_a.shape[-1] > 1:
            rows_a = rows_a[:, :-1]
            rows_b = rows_b[:, :-1]
        if rows_a.shape[-1] <= 1:
            return None
        if str(mode_a).startswith("sigmoid"):
            probs_a = torch.sigmoid(rows_a.detach())
            probs_b = torch.sigmoid(rows_b)
            foreground_mask = (
                probs_a.detach().max(dim=-1).values >= _SIGMOID_FOREGROUND_PROBABILITY_FLOOR
            )
            if not bool(foreground_mask.any()):
                return None
            probs_a = probs_a[foreground_mask]
            probs_b = probs_b[foreground_mask]
        else:
            probs_a = torch.softmax(rows_a.detach(), dim=-1)
            probs_b = torch.softmax(rows_b, dim=-1)
        return F.mse_loss(probs_b, probs_a)

    def _forward_ultralytics(self, model: object, images: list[torch.Tensor]) -> Any:
        from model_management.ultralytics_parity import (
            preprocess_bgr_images,
            rgb_tensor_to_bgr_uint8,
        )

        engine = getattr(model, "yolo", None) or getattr(model, "rtdetr", None)
        core = getattr(engine, "model", None)
        if engine is None or core is None:
            raise _TTASkip("logits_unavailable")
        images_bgr = [rgb_tensor_to_bgr_uint8(image) for image in images]
        _, model_input = preprocess_bgr_images(
            engine,
            images_bgr,
            conf=float(getattr(model, "confidence", 0.01)),
        )
        return core(model_input)

    def _forward_rfdetr(self, model: object, images: list[torch.Tensor]) -> Any | None:
        prepare_batch = getattr(model, "_prepare_batch", None)
        rfdetr = getattr(model, "rfdetr", None)
        rfdetr_context = getattr(rfdetr, "model", None)
        rfdetr_core = getattr(rfdetr_context, "model", None)
        if rfdetr is None and not callable(prepare_batch):
            return None
        if not callable(prepare_batch) or not callable(rfdetr_core):
            raise _TTASkip("logits_unavailable")
        batch_tensor, _ = prepare_batch(images)
        outputs = rfdetr_core(batch_tensor)
        return _normalize_rfdetr_tta_outputs(outputs)


def _normalize_rfdetr_tta_outputs(outputs: Any) -> dict[str, torch.Tensor]:
    if isinstance(outputs, dict):
        logits = outputs.get("pred_logits")
        boxes = outputs.get("pred_boxes")
        if (
            isinstance(logits, torch.Tensor)
            and isinstance(boxes, torch.Tensor)
            and _rfdetr_prefix_matches(logits, boxes)
            and _looks_like_rfdetr_boxes(boxes)
        ):
            return {
                "pred_logits": logits,
                "pred_boxes": boxes,
                "_tta_logit_mode": "sigmoid_bg_last",
            }
        raise _TTASkip("logits_unavailable")

    if hasattr(outputs, "logits") and hasattr(outputs, "pred_boxes"):
        logits = outputs.logits
        boxes = outputs.pred_boxes
        if (
            isinstance(logits, torch.Tensor)
            and isinstance(boxes, torch.Tensor)
            and _rfdetr_prefix_matches(logits, boxes)
            and _looks_like_rfdetr_boxes(boxes)
        ):
            return {
                "pred_logits": logits,
                "pred_boxes": boxes,
                "_tta_logit_mode": "sigmoid_bg_last",
            }
        raise _TTASkip("logits_unavailable")

    if isinstance(outputs, (tuple, list)):
        tensors = [value for value in outputs if isinstance(value, torch.Tensor)]
        for boxes in tensors:
            if not _looks_like_rfdetr_boxes(boxes):
                continue
            for logits in tensors:
                if logits is boxes:
                    continue
                if (
                    _rfdetr_prefix_matches(logits, boxes)
                    and int(logits.shape[-1]) != 4
                    and int(logits.shape[-1]) > 1
                ):
                    return {
                        "pred_logits": logits,
                        "pred_boxes": boxes,
                        "_tta_logit_mode": "sigmoid_bg_last",
                    }
        raise _TTASkip("logits_unavailable")

    raise _TTASkip("logits_unavailable")


def _looks_like_rfdetr_boxes(value: torch.Tensor) -> bool:
    return isinstance(value, torch.Tensor) and value.ndim >= 2 and int(value.shape[-1]) == 4


def _rfdetr_prefix_matches(logits: torch.Tensor, boxes: torch.Tensor) -> bool:
    return (
        isinstance(logits, torch.Tensor)
        and isinstance(boxes, torch.Tensor)
        and logits.ndim >= 2
        and boxes.ndim >= 2
        and tuple(logits.shape[:-1]) == tuple(boxes.shape[:-1])
    )


class SurgeonLocalTTAUpdater:
    def __init__(self, config: object, metrics_writer: object) -> None:
        self.config = config
        self.metrics = metrics_writer
        baseline_cfg = getattr(config, "baseline", None)
        self.method_cfg = getattr(baseline_cfg, "SURGEON", None)
        self.training_cfg = getattr(baseline_cfg, "training", None)
        quality_cfg = getattr(config, "sample_quality", None)
        drift_cfg = getattr(config, "window_drift", None)

        self.quality_classifier = EntropyQualityClassifier.from_config(quality_cfg)
        self.quality_mode = (
            str(
                getattr(self.method_cfg, "quality_mode", "output_only_when_no_boundary")
                or "output_only_when_no_boundary"
            )
            .strip()
            .lower()
        )
        self._output_entropy_window: deque[float] = deque(
            maxlen=max(
                1,
                int(
                    getattr(
                        getattr(quality_cfg, "output_entropy", None),
                        "window_size",
                        256,
                    )
                ),
            )
        )
        self._output_percentile = float(
            getattr(getattr(quality_cfg, "output_entropy", None), "percentile", 25.0)
        )
        self._output_warmup_samples = max(
            0,
            int(getattr(getattr(quality_cfg, "output_entropy", None), "warmup_samples", 20)),
        )
        configured_min_confidence = getattr(
            self.method_cfg,
            "min_detection_confidence",
            None,
        )
        if configured_min_confidence is None:
            configured_min_confidence = getattr(
                getattr(quality_cfg, "output_entropy", None),
                "min_detection_confidence",
                0.85,
            )
        self._output_min_confidence = max(
            0.0,
            min(
                1.0,
                float(configured_min_confidence),
            ),
        )
        self.drift_detector = WindowDriftDetector(
            window_size=int(getattr(drift_cfg, "window_size", 100)),
            min_window_size=int(getattr(drift_cfg, "min_window_size", 30)),
            low_quality_rate_threshold=float(getattr(drift_cfg, "low_quality_rate_threshold", 0.3)),
            persistence_windows=int(getattr(drift_cfg, "persistence_windows", 3)),
        )

        configured_training_frame_count = getattr(
            self.method_cfg,
            "training_frame_count",
            None,
        )
        if configured_training_frame_count is None:
            configured_training_frame_count = getattr(
                self.training_cfg,
                "training_frame_count",
                128,
            )
        self.training_frame_count = max(1, int(configured_training_frame_count))
        configured_train_sample_count = getattr(
            self.method_cfg,
            "train_sample_count",
            None,
        )
        if configured_train_sample_count is None:
            configured_train_sample_count = self.training_frame_count
        self.train_sample_count = max(
            1,
            min(int(configured_train_sample_count), int(self.training_frame_count)),
        )
        configured_num_epoch = getattr(self.method_cfg, "num_epoch", None)
        if configured_num_epoch is None:
            configured_num_epoch = getattr(self.training_cfg, "num_epoch", 1)
        self.num_epoch = max(1, int(configured_num_epoch))
        self.require_drift = bool(getattr(self.method_cfg, "require_drift", True))
        self.min_selected_logit_count = max(
            1,
            int(getattr(self.method_cfg, "min_selected_logit_count", 16)),
        )
        self.min_loss_improvement = max(
            0.0,
            float(getattr(self.method_cfg, "min_loss_improvement", 1.0e-4)),
        )
        self.consistency_weight = max(
            0.0,
            float(getattr(self.method_cfg, "consistency_weight", 0.0)),
        )
        self.entropy_margin_ratio = max(
            0.0,
            min(1.0, float(getattr(self.method_cfg, "entropy_margin_ratio", 0.4))),
        )
        self.adaptive_entropy_gate = getattr(
            self.method_cfg,
            "adaptive_entropy_gate",
            False,
        )
        self.max_entropy_margin_ratio = max(
            self.entropy_margin_ratio,
            min(
                1.0,
                max(
                    0.0,
                    float(
                        getattr(
                            self.method_cfg,
                            "max_entropy_margin_ratio",
                            0.7,
                        )
                    ),
                ),
            ),
        )
        self.trainable_scope = (
            str(getattr(self.method_cfg, "trainable_scope", "norm_affine") or "norm_affine")
            .strip()
            .lower()
        )
        self.batch_size = max(1, int(getattr(self.training_cfg, "batch_size", 32)))
        self.learning_rate = float(getattr(self.training_cfg, "learning_rate", 1.0e-3))
        self.weight_decay = float(getattr(self.training_cfg, "weight_decay", 0.0))
        self.optimizer_name = str(getattr(self.training_cfg, "optimizer_name", "adam"))

        self._edge = None
        self._buffer: deque[_BufferedSample] = deque(maxlen=self.training_frame_count)
        self._lock = threading.Lock()
        self._running_thread: threading.Thread | None = None
        self._pending_local_update: PendingLocalTTAUpdate | None = None
        self._closed = threading.Event()

    def attach_edge(self, edge) -> None:
        self._edge = edge
        logger.info(
            "[SURGEON] attached training_frame_count={} "
            "train_sample_count={} num_epoch={} batch_size={} quality_mode={} "
            "require_drift={} min_selected_logits={} entropy_margin={} "
            "adaptive_entropy_gate={} max_entropy_margin={}",
            self.training_frame_count,
            self.train_sample_count,
            self.num_epoch,
            self.batch_size,
            self.quality_mode,
            self.require_drift,
            self.min_selected_logit_count,
            self.entropy_margin_ratio,
            self.adaptive_entropy_gate,
            self.max_entropy_margin_ratio,
        )
        self.metrics.record(
            "surgeon_tta_config",
            training_frame_count=int(self.training_frame_count),
            train_sample_count=int(self.train_sample_count),
            num_epoch=int(self.num_epoch),
            mini_batch_size=int(self.batch_size),
            trainable_scope=str(self.trainable_scope),
            consistency_weight=float(self.consistency_weight),
            min_loss_improvement=float(self.min_loss_improvement),
            min_selected_logit_count=int(self.min_selected_logit_count),
            entropy_margin_ratio=float(self.entropy_margin_ratio),
            adaptive_entropy_gate=bool(self.adaptive_entropy_gate),
            max_entropy_margin_ratio=float(self.max_entropy_margin_ratio),
        )

    def observe_sample(
        self,
        frame,
        frame_index: int,
        task,
        artifacts: dict[str, Any],
        latency_ms: float | None,
    ) -> None:
        if self._closed.is_set():
            return
        quality = self._classify_sample(
            artifacts=artifacts,
            model_name=str(getattr(self.config, "lightweight", "") or ""),
        )
        drift = self.drift_detector.update(quality)
        self._record_quality(frame_index, quality)
        self.metrics.record(
            "drift_window_summary",
            frame_id=int(frame_index),
            window_id=drift.window_id,
            drift_detected=bool(drift.drift_detected),
            drift_score=float(drift.drift_score),
            low_quality_rate=float(drift.low_quality_rate),
            drift_reasons=list(drift.drift_reasons),
        )
        if quality.quality_bucket != LOW_QUALITY:
            return
        sample = _BufferedSample(
            frame_id=int(frame_index),
            frame=np.ascontiguousarray(frame).copy(),
            artifacts=dict(artifacts or {}),
            latency_ms=latency_ms,
        )
        with self._lock:
            self._buffer.append(sample)
            if len(self._buffer) < self.training_frame_count:
                return
            if self.require_drift and not drift.drift_detected:
                return
            if self._running_thread is not None and self._running_thread.is_alive():
                return
            if self._pending_local_update is not None:
                return
            selected = list(self._buffer)[-int(self.train_sample_count) :]
            buffered_count = len(self._buffer)
            # Consume this trigger window. Samples observed while the shadow
            # model trains may form a later window, but a successful apply
            # discards them because they came from the old live model.
            self._buffer.clear()
            logger.info(
                "[SURGEON] local TTA triggered: low_quality={} "
                "training_frame_count={} train_sample_count={} "
                "mini_batch_size={} trigger_frame={}",
                buffered_count,
                self.training_frame_count,
                len(selected),
                self.batch_size,
                int(frame_index),
            )
            self.metrics.record(
                "surgeon_tta_triggered",
                frame_id=int(frame_index),
                low_quality_sample_count=buffered_count,
                batch_size=int(self.batch_size),
                training_frame_count=int(self.training_frame_count),
                train_sample_count=len(selected),
                drift_detected=bool(drift.drift_detected),
                drift_score=float(drift.drift_score),
            )
            self._running_thread = threading.Thread(
                target=self._run_tta_task,
                args=(selected, int(frame_index)),
                name="pure-edge-surgeon-tta",
                daemon=True,
            )
            self._running_thread.start()

    def close(self) -> None:
        self._closed.set()
        self.wait_for_idle(timeout=10.0)
        self.try_apply_pending_update()

    def wait_for_idle(self, *, timeout: float = 10.0) -> bool:
        thread = self._running_thread
        if thread is not None and thread.is_alive() and threading.current_thread() is not thread:
            thread.join(timeout=max(0.0, float(timeout)))
        thread = self._running_thread
        return thread is None or not thread.is_alive()

    def _classify_sample(
        self,
        *,
        artifacts: dict[str, Any],
        model_name: str,
    ) -> EntropyQualityStats:
        prediction = _prediction_from_artifacts(artifacts)
        boundary_payload = _boundary_payload_from_artifacts(artifacts)
        if boundary_payload is not None or self.quality_mode != "output_only_when_no_boundary":
            return self.quality_classifier.classify(
                prediction,
                boundary_payload,
                model_name,
                "pure-edge-local",
                "pure-edge-local",
            )
        return self._classify_output_only(prediction)

    def _classify_output_only(self, prediction: dict[str, Any]) -> EntropyQualityStats:
        output_entropy = _output_entropy(prediction)
        threshold = _percentile(list(self._output_entropy_window), self._output_percentile)
        warmed = len(self._output_entropy_window) >= self._output_warmup_samples
        if threshold is None and self._output_warmup_samples <= 0:
            threshold = 1.0
            warmed = True
        output_reliable = (
            output_entropy is not None
            and threshold is not None
            and warmed
            and float(output_entropy) <= float(threshold)
        )
        if output_entropy is not None:
            self._output_entropy_window.append(float(output_entropy))

        confidence = _confidence(prediction)
        output_confident = self._output_min_confidence <= 0.0 or (
            confidence is not None and float(confidence) >= self._output_min_confidence
        )
        empty = _predictions_empty(prediction)
        trusted = not empty and bool(output_reliable) and bool(output_confident)
        reasons: list[str] = ["output_only_no_boundary"]
        if empty:
            reasons.append("empty_predictions")
        if output_entropy is None:
            reasons.append("output_entropy_unavailable")
        elif not warmed:
            reasons.append("output_entropy_warmup")
        elif not output_reliable:
            reasons.append("output_entropy_high")
        if not output_confident:
            reasons.append(
                "output_confidence_unavailable" if confidence is None else "output_confidence_low"
            )
        if trusted:
            reasons.append("trusted_edge_pseudo_label")
        return EntropyQualityStats(
            output_entropy=output_entropy,
            output_entropy_threshold=threshold,
            output_confidence=confidence,
            output_confidence_threshold=self._output_min_confidence,
            output_confident=bool(output_confident),
            feature_entropy=None,
            feature_entropy_mean=None,
            feature_entropy_std=None,
            feature_entropy_deviation=None,
            feature_deviation_threshold=0.0,
            output_reliable=bool(output_reliable),
            feature_normal=True,
            edge_pseudo_label_trusted=bool(trusted),
            quality=HIGH_QUALITY if trusted else LOW_QUALITY,
            reason=";".join(reasons),
        )

    def _record_quality(self, frame_index: int, quality: EntropyQualityStats) -> None:
        self.metrics.record(
            "sample_quality_summary",
            frame_id=int(frame_index),
            quality=quality.quality,
            reason=quality.reason,
            output_entropy=quality.output_entropy,
            output_entropy_threshold=quality.output_entropy_threshold,
            output_confidence=quality.output_confidence,
            output_confidence_threshold=quality.output_confidence_threshold,
            output_reliable=bool(quality.output_reliable),
            feature_entropy=quality.feature_entropy,
            feature_normal=bool(quality.feature_normal),
            in_drift_window=bool(quality.in_drift_window),
            window_id=quality.window_id,
        )

    def _run_tta_task(self, samples: list[_BufferedSample], trigger_frame_id: int) -> None:
        started = time.perf_counter()
        self.metrics.record(
            "surgeon_tta_started",
            frame_id=int(trigger_frame_id),
            low_quality_sample_count=len(samples),
            batch_size=len(samples),
            num_epoch=self.num_epoch,
        )
        try:
            update = self._execute_tta(samples, trigger_frame_id, started)
        except _TTASkip as exc:
            logger.info(
                "[SURGEON][Train] skipped reason={} low_quality_sample_count={}",
                exc.reason,
                len(samples),
            )
            self.metrics.record(
                "surgeon_tta_skipped",
                frame_id=int(trigger_frame_id),
                reason=exc.reason,
                low_quality_sample_count=len(samples),
                **exc.details,
            )
        except Exception as exc:  # noqa: BLE001 - metrics must capture runtime failures.
            logger.warning("[SURGEON] local TTA failed: {}", exc)
            self.metrics.record(
                "surgeon_tta_failed",
                frame_id=int(trigger_frame_id),
                message=str(exc),
                low_quality_sample_count=len(samples),
            )
        else:
            queued = False
            with self._lock:
                if self._pending_local_update is None:
                    self._pending_local_update = update
                    queued = True
            if queued:
                logger.info(
                    "[SURGEON] shadow training done: final_loss={:.6f} "
                    "pending_apply=true",
                    float(update.loss),
                )
                self.metrics.record(
                    "surgeon_tta_local_update_pending",
                    frame_id=int(trigger_frame_id),
                    batch_size=int(update.batch_size),
                    num_epoch=int(update.num_epoch),
                    initial_loss=float(update.initial_loss),
                    loss=float(update.loss),
                    initial_selected_logit_count=int(
                        update.initial_selected_logit_count
                    ),
                    selected_logit_count=int(update.selected_logit_count),
                    **_prefixed_gate_stats(
                        update.initial_gate_stats,
                        prefix="initial_",
                    ),
                    **update.gate_stats,
                    model_version_before=str(update.model_version_before),
                )
            else:
                self.metrics.record(
                    "surgeon_tta_failed",
                    frame_id=int(trigger_frame_id),
                    message="pending_local_update_exists",
                    low_quality_sample_count=len(samples),
                )
        finally:
            with self._lock:
                self._running_thread = None

    def _execute_tta(
        self,
        samples: list[_BufferedSample],
        trigger_frame_id: int,
        started_perf: float,
    ) -> PendingLocalTTAUpdate:
        if self._edge is None:
            raise _TTASkip("edge_unavailable")
        detector = getattr(self._edge, "small_object_detection", None)
        if detector is None:
            raise _TTASkip("detector_unavailable")
        live_model = getattr(detector, "model", detector)
        model_lock = getattr(detector, "model_lock", None)
        if model_lock is None:
            raise _TTASkip("model_lock_unavailable")

        self.metrics.record(
            "surgeon_tta_shadow_snapshot_started",
            frame_id=int(trigger_frame_id),
            low_quality_sample_count=len(samples),
            batch_size=len(samples),
        )
        with model_lock:
            snapshot = self._snapshot_live_model_locked(detector, live_model)
        self.metrics.record(
            "surgeon_tta_shadow_snapshot_done",
            frame_id=int(trigger_frame_id),
            snapshot_lock_ms=float(snapshot["snapshot_lock_ms"]),
            model_version_before=str(snapshot["model_version_before"]),
            model_class=str(snapshot["model_class"]),
            trainable_model_class=str(snapshot["trainable_model_class"]),
        )

        shadow_model = self._build_shadow_training_model(detector, snapshot)
        adapter = TTADetectionAdapter(
            detector,
            model_override=shadow_model,
            entropy_margin_ratio=self.entropy_margin_ratio,
            adaptive_entropy_gate=self.adaptive_entropy_gate,
            max_entropy_margin_ratio=self.max_entropy_margin_ratio,
            min_selected_logit_count=self.min_selected_logit_count,
        )
        batch = adapter.build_batch([sample.frame for sample in samples])
        train_result = self._train_shadow_model(
            adapter=adapter,
            batch=batch,
            trigger_frame_id=trigger_frame_id,
            model_version_before=str(snapshot["model_version_before"]),
        )
        return PendingLocalTTAUpdate(
            trigger_frame_id=int(trigger_frame_id),
            trained_state_dict=train_result["trained_state_dict"],
            model_version_before=str(snapshot["model_version_before"]),
            snapshot_lock_ms=float(snapshot["snapshot_lock_ms"]),
            batch_size=int(train_result["batch_size"]),
            num_epoch=int(self.num_epoch),
            initial_loss=float(train_result["initial_loss"]),
            loss=float(train_result["loss"]),
            initial_selected_logit_count=int(
                train_result["initial_selected_logit_count"]
            ),
            selected_logit_count=int(train_result["selected_logit_count"]),
            initial_gate_stats=dict(train_result["initial_gate_stats"]),
            gate_stats=dict(train_result["gate_stats"]),
            trainable_param_count=int(train_result["trainable_param_count"]),
            low_quality_sample_count=len(samples),
            started_perf=float(started_perf),
            shadow_train_ms=float(train_result["shadow_train_ms"]),
            live_training_mode=bool(snapshot["training"]),
            live_module_training_state=dict(snapshot["module_training_state"]),
        )

    def try_apply_pending_update(self) -> bool:
        if self._edge is None:
            return False
        with self._lock:
            update = self._pending_local_update
        if update is None:
            return False
        detector = getattr(self._edge, "small_object_detection", None)
        if detector is None:
            self._discard_pending_update_failed(update, "detector_unavailable")
            return False
        model_lock = getattr(detector, "model_lock", None)
        if model_lock is None:
            self._discard_pending_update_failed(update, "model_lock_unavailable")
            return False
        if not model_lock.acquire(blocking=False):
            return False
        with self._lock:
            if self._pending_local_update is not update:
                model_lock.release()
                return False
        try:
            live_model = getattr(detector, "model", detector)
            model_version_after, apply_lock_ms = self._apply_shadow_update_locked(
                detector,
                live_model,
                update.trained_state_dict,
                update.model_version_before,
                update.live_training_mode,
                update.live_module_training_state,
            )
        except Exception as exc:  # noqa: BLE001 - preserve live inference and record metrics.
            self.metrics.record(
                "surgeon_tta_failed",
                frame_id=int(update.trigger_frame_id),
                message=str(exc),
                low_quality_sample_count=int(update.low_quality_sample_count),
                model_version_before=str(update.model_version_before),
            )
            with self._lock:
                if self._pending_local_update is update:
                    self._pending_local_update = None
            return False
        finally:
            model_lock.release()

        duration_ms = (time.perf_counter() - update.started_perf) * 1000.0
        logger.info(
            "[SURGEON] local update applied: model_version={} -> {} "
            "apply_lock_ms={:.3f}",
            update.model_version_before,
            model_version_after,
            apply_lock_ms,
        )
        self.metrics.record(
            "surgeon_tta_done",
            frame_id=int(update.trigger_frame_id),
            low_quality_sample_count=int(update.low_quality_sample_count),
            batch_size=int(update.batch_size),
            num_epoch=int(update.num_epoch),
            loss=float(update.loss),
            initial_loss=float(update.initial_loss),
            initial_selected_logit_count=int(update.initial_selected_logit_count),
            selected_logit_count=int(update.selected_logit_count),
            **_prefixed_gate_stats(
                update.initial_gate_stats,
                prefix="initial_",
            ),
            **update.gate_stats,
            duration_ms=float(duration_ms),
            shadow_train_ms=float(update.shadow_train_ms),
            trainable_param_count=int(update.trainable_param_count),
            model_version_before=str(update.model_version_before),
            model_version_after=str(model_version_after),
            shadow_training=True,
            live_model_lock_held_during_training=False,
            snapshot_lock_ms=float(update.snapshot_lock_ms),
            apply_lock_ms=float(apply_lock_ms),
        )
        self.metrics.record(
            "local_model_update_applied",
            frame_id=int(update.trigger_frame_id),
            model_version_before=str(update.model_version_before),
            model_version_after=str(model_version_after),
            apply_lock_ms=float(apply_lock_ms),
        )
        with self._lock:
            if self._pending_local_update is update:
                self._pending_local_update = None
            self._buffer.clear()
        self.drift_detector.reset()
        self._output_entropy_window.clear()
        return True

    def _discard_pending_update_failed(
        self,
        update: PendingLocalTTAUpdate,
        reason: str,
    ) -> None:
        self.metrics.record(
            "surgeon_tta_failed",
            frame_id=int(update.trigger_frame_id),
            message=str(reason),
            low_quality_sample_count=int(update.low_quality_sample_count),
            model_version_before=str(update.model_version_before),
        )
        with self._lock:
            if self._pending_local_update is update:
                self._pending_local_update = None

    def _snapshot_live_model_locked(self, detector: object, live_model: object) -> dict[str, Any]:
        started = time.perf_counter()
        trainable_model = unwrap_trainable_module(
            live_model,
            model_name=str(getattr(detector, "model_name", "")),
        )
        if not hasattr(trainable_model, "state_dict"):
            raise _TTASkip("model_state_unavailable")
        try:
            num_classes = int(getattr(live_model, "num_classes"))
        except (AttributeError, TypeError, ValueError):
            num_classes = None
        if num_classes is not None and num_classes <= 0:
            num_classes = None
        state_dict = _clone_state_dict_to_cpu(trainable_model.state_dict())
        snapshot_lock_ms = (time.perf_counter() - started) * 1000.0
        return {
            "live_model_ref": live_model,
            "model_name": str(getattr(detector, "model_name", "") or ""),
            "num_classes": num_classes,
            "model_class": type(live_model).__name__,
            "trainable_model_class": type(trainable_model).__name__,
            "device": _model_device(trainable_model),
            "state_dict": state_dict,
            "model_version_before": str(getattr(self._edge, "model_version", "0") or "0"),
            "training": bool(getattr(trainable_model, "training", False)),
            "module_training_state": _module_training_state(trainable_model),
            "snapshot_lock_ms": snapshot_lock_ms,
        }

    def _build_shadow_training_model(
        self,
        detector: object,
        snapshot: dict[str, Any],
    ) -> torch.nn.Module:
        live_model = snapshot["live_model_ref"]
        try:
            shadow_model = copy.deepcopy(live_model)
        except Exception as deepcopy_exc:  # noqa: BLE001 - fallback to model zoo below.
            shadow_model = self._build_shadow_from_model_zoo(detector, snapshot, deepcopy_exc)
        shadow_trainable = unwrap_trainable_module(
            shadow_model,
            model_name=str(snapshot.get("model_name", "")),
        )
        training_device = self._resolve_training_device(snapshot)
        if isinstance(shadow_model, torch.nn.Module):
            shadow_model.to(training_device)
        if isinstance(shadow_trainable, torch.nn.Module):
            shadow_trainable.to(training_device)
        shadow_trainable.load_state_dict(
            _state_dict_to_device(snapshot["state_dict"], training_device),
            strict=True,
        )
        return shadow_model

    def _build_shadow_from_model_zoo(
        self,
        detector: object,
        snapshot: dict[str, Any],
        deepcopy_exc: Exception,
    ) -> torch.nn.Module:
        model_name = str(snapshot.get("model_name", "") or "")
        if not model_name:
            raise RuntimeError(
                "shadow model deepcopy failed and detector model_name is unavailable"
            ) from deepcopy_exc
        detector_config = getattr(detector, "config", None)
        build_kwargs: dict[str, Any] = {}
        num_classes = snapshot.get("num_classes")
        if num_classes is not None:
            build_kwargs["num_classes"] = int(num_classes)
        try:
            if get_model_family(model_name) == "tinynext":
                configured_input_size = getattr(detector_config, "tinynext_input_size", None)
                if configured_input_size is not None:
                    build_kwargs["tinynext_input_size"] = int(configured_input_size)
        except Exception:
            pass
        live_model = snapshot.get("live_model_ref")
        confidence = float(getattr(live_model, "confidence", 0.01))
        return build_detection_model(
            model_name,
            pretrained=False,
            device=self._resolve_training_device(snapshot),
            confidence=confidence,
            **build_kwargs,
        )

    def _train_shadow_model(
        self,
        *,
        adapter: TTADetectionAdapter,
        batch: list[torch.Tensor],
        trigger_frame_id: int,
        model_version_before: str,
    ) -> dict[str, Any]:
        trainable_model = adapter.trainable_model
        training_device = _model_device(trainable_model)
        batch = list(batch)
        total_sample_count = len(batch)
        mini_batch_size = max(1, int(self.batch_size))
        mini_batches = [
            batch[index : index + mini_batch_size]
            for index in range(0, total_sample_count, mini_batch_size)
        ]
        previous_mode = bool(getattr(trainable_model, "training", False))
        module_training_state = _module_training_state(trainable_model)
        grad_state = _parameter_grad_state(trainable_model)
        batch_norm_tracking_state = _batch_norm_tracking_state(trainable_model)
        train_started = time.perf_counter()
        self.metrics.record(
            "surgeon_tta_shadow_train_started",
            frame_id=int(trigger_frame_id),
            batch_size=int(total_sample_count),
            mini_batch_size=int(mini_batch_size),
            num_epoch=int(self.num_epoch),
            model_version_before=str(model_version_before),
            min_selected_logit_count=int(self.min_selected_logit_count),
            entropy_margin_ratio=float(self.entropy_margin_ratio),
            adaptive_entropy_gate=bool(self.adaptive_entropy_gate),
            max_entropy_margin_ratio=float(self.max_entropy_margin_ratio),
        )
        logger.info(
            "[SURGEON] shadow training started: samples={} "
            "mini_batch_size={} epochs={}",
            int(total_sample_count),
            int(mini_batch_size),
            int(self.num_epoch),
        )
        try:
            self._select_trainable_parameters(trainable_model)
            trainable_params = [
                param for param in trainable_model.parameters() if param.requires_grad
            ]
            trainable_param_count = sum(int(param.numel()) for param in trainable_params)
            if not trainable_params:
                raise _TTASkip("no_trainable_parameters")
            optimizer = self._make_optimizer(trainable_params)
            _set_batch_norm_tracking(trainable_model, enabled=False)
            if hasattr(trainable_model, "train"):
                trainable_model.train(True)
            optimizer.zero_grad(set_to_none=True)
            initial_objective = self._evaluate_shadow_objective(
                adapter=adapter,
                mini_batches=mini_batches,
                training_device=training_device,
            )
            losses: list[float] = []
            for epoch_index in range(self.num_epoch):
                epoch_started = time.perf_counter()
                epoch = epoch_index + 1
                epoch_loss_values: list[float] = []
                epoch_entropy_values: list[float] = []
                epoch_consistency_values: list[float] = []
                epoch_weighted_consistency_values: list[float] = []
                epoch_logit_count = 0
                epoch_foreground_logit_count = 0
                epoch_strict_selected_logit_count = 0
                epoch_max_entropy_candidate_count = 0
                epoch_selected_logit_count = 0
                epoch_effective_entropy_thresholds: list[float] = []
                epoch_adaptive_entropy_gate_batch_count = 0
                epoch_consistency_batch_count = 0
                for mini_batch in mini_batches:
                    device_batch = _move_batch_to_device(mini_batch, training_device)
                    optimizer.zero_grad(set_to_none=True)
                    outputs = adapter.forward_tta_outputs(device_batch)
                    entropy_loss, loss_stats = adapter.entropy_loss(outputs)
                    self._require_reliable_logits(loss_stats)
                    loss = entropy_loss
                    consistency_loss_value = 0.0
                    weighted_consistency_loss_value = 0.0
                    if self.consistency_weight > 0.0:
                        augmented = adapter.forward_tta_outputs(device_batch, augment=True)
                        consistency = adapter.consistency_loss(outputs, augmented)
                        if consistency is not None:
                            epoch_consistency_batch_count += 1
                            consistency_loss_value = float(consistency.detach().item())
                            weighted_consistency = self.consistency_weight * consistency
                            weighted_consistency_loss_value = float(
                                weighted_consistency.detach().item()
                            )
                            loss = loss + weighted_consistency
                    loss.backward()
                    optimizer.step()
                    epoch_loss_values.append(float(loss.detach().item()))
                    epoch_entropy_values.append(float(entropy_loss.detach().item()))
                    epoch_consistency_values.append(consistency_loss_value)
                    epoch_weighted_consistency_values.append(
                        weighted_consistency_loss_value
                    )
                    epoch_logit_count += int(loss_stats.get("logit_count", 0))
                    epoch_foreground_logit_count += int(
                        loss_stats.get("foreground_logit_count", 0)
                    )
                    epoch_strict_selected_logit_count += int(
                        loss_stats.get("strict_selected_logit_count", 0)
                    )
                    epoch_max_entropy_candidate_count += int(
                        loss_stats.get("max_entropy_candidate_count", 0)
                    )
                    epoch_selected_logit_count += int(
                        loss_stats.get("selected_logit_count", 0)
                    )
                    effective_threshold = _finite_float(
                        loss_stats.get("effective_entropy_threshold")
                    )
                    if effective_threshold is not None:
                        epoch_effective_entropy_thresholds.append(effective_threshold)
                    if bool(loss_stats.get("adaptive_entropy_gate_used", False)):
                        epoch_adaptive_entropy_gate_batch_count += 1
                loss_value = float(np.mean(epoch_loss_values)) if epoch_loss_values else 0.0
                entropy_loss_value = (
                    float(np.mean(epoch_entropy_values)) if epoch_entropy_values else 0.0
                )
                consistency_loss_value = (
                    float(np.mean(epoch_consistency_values))
                    if epoch_consistency_values
                    else 0.0
                )
                weighted_consistency_loss_value = (
                    float(np.mean(epoch_weighted_consistency_values))
                    if epoch_weighted_consistency_values
                    else 0.0
                )
                epoch_ms = (time.perf_counter() - epoch_started) * 1000.0
                losses.append(loss_value)
                logger.info(
                    "[SURGEON][Train] epoch={}/{} loss={:.6f} "
                    "entropy_loss={:.6f} consistency_loss={:.6f} samples={} "
                    "mini_batch_size={} selected_logits={}/{}/{}/{}/{} "
                    "effective_entropy_threshold={:.4f} adaptive_batches={} "
                    "model_version={} "
                    "epoch_ms={:.3f}",
                    epoch,
                    self.num_epoch,
                    loss_value,
                    entropy_loss_value,
                    consistency_loss_value,
                    total_sample_count,
                    mini_batch_size,
                    epoch_selected_logit_count,
                    epoch_strict_selected_logit_count,
                    epoch_max_entropy_candidate_count,
                    epoch_foreground_logit_count,
                    epoch_logit_count,
                    max(epoch_effective_entropy_thresholds, default=0.0),
                    epoch_adaptive_entropy_gate_batch_count,
                    model_version_before,
                    epoch_ms,
                )
                self.metrics.record(
                    "surgeon_tta_epoch",
                    frame_id=int(trigger_frame_id),
                    epoch=int(epoch),
                    total_epochs=int(self.num_epoch),
                    loss=loss_value,
                    entropy_loss=entropy_loss_value,
                    consistency_loss=consistency_loss_value,
                    weighted_consistency_loss=weighted_consistency_loss_value,
                    batch_size=int(total_sample_count),
                    mini_batch_size=int(mini_batch_size),
                    logit_count=int(epoch_logit_count),
                    foreground_logit_count=int(epoch_foreground_logit_count),
                    strict_selected_logit_count=int(
                        epoch_strict_selected_logit_count
                    ),
                    max_entropy_candidate_count=int(
                        epoch_max_entropy_candidate_count
                    ),
                    selected_logit_count=int(epoch_selected_logit_count),
                    effective_entropy_threshold=float(
                        max(epoch_effective_entropy_thresholds, default=0.0)
                    ),
                    adaptive_entropy_gate=bool(self.adaptive_entropy_gate),
                    adaptive_entropy_gate_used=bool(
                        epoch_adaptive_entropy_gate_batch_count
                    ),
                    adaptive_entropy_gate_batch_count=int(
                        epoch_adaptive_entropy_gate_batch_count
                    ),
                    entropy_margin_ratio=float(self.entropy_margin_ratio),
                    max_entropy_margin_ratio=float(self.max_entropy_margin_ratio),
                    required_selected_logit_count=int(
                        self.min_selected_logit_count
                    ),
                    consistency_batch_count=int(epoch_consistency_batch_count),
                    model_version=str(model_version_before),
                    epoch_ms=float(epoch_ms),
                )
            optimizer.zero_grad(set_to_none=True)
            final_objective = self._evaluate_shadow_objective(
                adapter=adapter,
                mini_batches=mini_batches,
                training_device=training_device,
            )
            initial_loss = float(initial_objective["loss"])
            final_loss = float(final_objective["loss"])
            accepted = math.isfinite(final_loss) and (
                final_loss < initial_loss - self.min_loss_improvement
            )
            if not accepted:
                self.metrics.record(
                    "surgeon_tta_rejected",
                    frame_id=int(trigger_frame_id),
                    reason="objective_not_improved",
                    initial_loss=initial_loss,
                    final_loss=final_loss,
                    required_improvement=float(self.min_loss_improvement),
                    initial_selected_logit_count=int(
                        initial_objective["selected_logit_count"]
                    ),
                    selected_logit_count=int(final_objective["selected_logit_count"]),
                    **_prefixed_gate_stats(
                        _gate_stats_payload(initial_objective),
                        prefix="initial_",
                    ),
                    **_gate_stats_payload(final_objective),
                    model_version_before=str(model_version_before),
                )
                raise _TTASkip("objective_not_improved")
            shadow_train_ms = (time.perf_counter() - train_started) * 1000.0
            self.metrics.record(
                "surgeon_tta_shadow_train_done",
                frame_id=int(trigger_frame_id),
                batch_size=int(total_sample_count),
                mini_batch_size=int(mini_batch_size),
                num_epoch=int(self.num_epoch),
                initial_loss=initial_loss,
                loss=float(final_loss),
                training_loss=float(losses[-1] if losses else initial_loss),
                initial_selected_logit_count=int(
                    initial_objective["selected_logit_count"]
                ),
                selected_logit_count=int(final_objective["selected_logit_count"]),
                foreground_logit_count=int(final_objective["foreground_logit_count"]),
                logit_count=int(final_objective["logit_count"]),
                **_prefixed_gate_stats(
                    _gate_stats_payload(initial_objective),
                    prefix="initial_",
                ),
                **_gate_stats_payload(final_objective),
                shadow_train_ms=float(shadow_train_ms),
                model_version_before=str(model_version_before),
                trainable_param_count=int(trainable_param_count),
            )
            return {
                "batch_size": total_sample_count,
                "initial_loss": initial_loss,
                "loss": final_loss,
                "initial_selected_logit_count": int(
                    initial_objective["selected_logit_count"]
                ),
                "selected_logit_count": int(final_objective["selected_logit_count"]),
                "initial_gate_stats": _gate_stats_payload(initial_objective),
                "gate_stats": _gate_stats_payload(final_objective),
                "trainable_param_count": trainable_param_count,
                "shadow_train_ms": shadow_train_ms,
                "trained_state_dict": _clone_state_dict_to_cpu(trainable_model.state_dict()),
            }
        finally:
            _clear_gradients(trainable_model)
            _restore_batch_norm_tracking(trainable_model, batch_norm_tracking_state)
            _restore_parameter_grad_state(trainable_model, grad_state)
            _restore_module_training_state(
                trainable_model,
                module_training_state,
                previous_mode,
            )

    def _evaluate_shadow_objective(
        self,
        *,
        adapter: TTADetectionAdapter,
        mini_batches: list[list[torch.Tensor]],
        training_device: torch.device,
    ) -> dict[str, float | int]:
        loss_values: list[float] = []
        entropy_values: list[float] = []
        consistency_values: list[float] = []
        logit_count = 0
        foreground_logit_count = 0
        strict_selected_logit_count = 0
        max_entropy_candidate_count = 0
        selected_logit_count = 0
        effective_entropy_thresholds: list[float] = []
        adaptive_entropy_gate_batch_count = 0
        consistency_batch_count = 0
        for mini_batch in mini_batches:
            device_batch = _move_batch_to_device(mini_batch, training_device)
            outputs = adapter.forward_tta_outputs(device_batch)
            entropy_loss, loss_stats = adapter.entropy_loss(outputs)
            self._require_reliable_logits(loss_stats)
            loss = entropy_loss
            consistency_value = 0.0
            if self.consistency_weight > 0.0:
                augmented = adapter.forward_tta_outputs(device_batch, augment=True)
                consistency = adapter.consistency_loss(outputs, augmented)
                if consistency is not None:
                    consistency_batch_count += 1
                    consistency_value = float(consistency.detach().item())
                    loss = loss + self.consistency_weight * consistency
            loss_values.append(float(loss.detach().item()))
            entropy_values.append(float(entropy_loss.detach().item()))
            consistency_values.append(consistency_value)
            logit_count += int(loss_stats.get("logit_count", 0))
            foreground_logit_count += int(loss_stats.get("foreground_logit_count", 0))
            strict_selected_logit_count += int(
                loss_stats.get("strict_selected_logit_count", 0)
            )
            max_entropy_candidate_count += int(
                loss_stats.get("max_entropy_candidate_count", 0)
            )
            selected_logit_count += int(loss_stats.get("selected_logit_count", 0))
            effective_threshold = _finite_float(
                loss_stats.get("effective_entropy_threshold")
            )
            if effective_threshold is not None:
                effective_entropy_thresholds.append(effective_threshold)
            if bool(loss_stats.get("adaptive_entropy_gate_used", False)):
                adaptive_entropy_gate_batch_count += 1
        return {
            "loss": float(np.mean(loss_values)) if loss_values else float("inf"),
            "entropy_loss": (
                float(np.mean(entropy_values)) if entropy_values else float("inf")
            ),
            "consistency_loss": (
                float(np.mean(consistency_values)) if consistency_values else 0.0
            ),
            "logit_count": int(logit_count),
            "foreground_logit_count": int(foreground_logit_count),
            "strict_selected_logit_count": int(strict_selected_logit_count),
            "max_entropy_candidate_count": int(max_entropy_candidate_count),
            "selected_logit_count": int(selected_logit_count),
            "effective_entropy_threshold": float(
                max(effective_entropy_thresholds, default=0.0)
            ),
            "adaptive_entropy_gate_used": bool(
                adaptive_entropy_gate_batch_count
            ),
            "adaptive_entropy_gate_batch_count": int(
                adaptive_entropy_gate_batch_count
            ),
            "consistency_batch_count": int(consistency_batch_count),
        }

    def _require_reliable_logits(self, loss_stats: dict[str, Any]) -> None:
        selected_count = int(loss_stats.get("selected_logit_count", 0))
        if selected_count < self.min_selected_logit_count:
            raise _TTASkip(
                "insufficient_reliable_logits",
                required_selected_logit_count=int(self.min_selected_logit_count),
                actual_selected_logit_count=selected_count,
                strict_selected_logit_count=int(
                    loss_stats.get("strict_selected_logit_count", 0)
                ),
                max_entropy_candidate_count=int(
                    loss_stats.get("max_entropy_candidate_count", selected_count)
                ),
                entropy_margin_ratio=float(self.entropy_margin_ratio),
                max_entropy_margin_ratio=float(self.max_entropy_margin_ratio),
                adaptive_entropy_gate=bool(self.adaptive_entropy_gate),
            )

    def _apply_shadow_update_locked(
        self,
        detector: object,
        live_model: object,
        trained_state_dict: dict[str, Any],
        model_version_before: str,
        live_training_mode: bool,
        live_module_training_state: dict[str, bool],
    ) -> tuple[str, float]:
        del detector
        started = time.perf_counter()
        trainable_model = unwrap_trainable_module(live_model)
        current_version = str(getattr(self._edge, "model_version", "0") or "0")
        if current_version != str(model_version_before):
            raise RuntimeError(
                "live model version changed before local SURGEON apply: "
                f"{current_version} != {model_version_before}"
            )
        live_state = trainable_model.state_dict()
        _validate_state_dict_compatible(live_state, trained_state_dict)
        trainable_model.load_state_dict(
            _state_dict_to_device(trained_state_dict, _model_device(trainable_model)),
            strict=True,
        )
        _clear_gradients(trainable_model)
        _restore_module_training_state(
            trainable_model,
            live_module_training_state,
            bool(live_training_mode),
        )
        model_version_after = _next_surgeon_version(model_version_before)
        self._edge.model_version = model_version_after
        apply_lock_ms = (time.perf_counter() - started) * 1000.0
        return model_version_after, apply_lock_ms

    def _resolve_training_device(self, snapshot: dict[str, Any]) -> torch.device:
        configured = str(getattr(self.training_cfg, "device", "auto") or "auto").strip().lower()
        if configured and configured != "auto":
            return torch.device(configured)
        device = snapshot.get("device")
        return device if isinstance(device, torch.device) else torch.device(str(device or "cpu"))

    def _select_trainable_parameters(self, model: torch.nn.Module) -> None:
        for param in model.parameters():
            param.requires_grad_(False)
        if self.trainable_scope == "norm_affine":
            _enable_norm_affine_parameters(model)

    def _make_optimizer(self, params: list[torch.nn.Parameter]):
        name = self.optimizer_name.strip().lower()
        if name == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        return torch.optim.Adam(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


def _gate_stats_payload(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "strict_selected_logit_count": int(
            stats.get("strict_selected_logit_count", 0)
        ),
        "max_entropy_candidate_count": int(
            stats.get("max_entropy_candidate_count", 0)
        ),
        "effective_entropy_threshold": float(
            stats.get("effective_entropy_threshold", 0.0)
        ),
        "adaptive_entropy_gate_used": bool(
            stats.get("adaptive_entropy_gate_used", False)
        ),
        "adaptive_entropy_gate_batch_count": int(
            stats.get("adaptive_entropy_gate_batch_count", 0)
        ),
    }


def _prefixed_gate_stats(
    stats: dict[str, Any],
    *,
    prefix: str,
) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in stats.items()}


def _clone_state_dict_to_cpu(state_dict: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cloned[str(key)] = value.detach().clone().cpu()
        else:
            cloned[str(key)] = copy.deepcopy(value)
    return cloned


def _state_dict_to_device(state_dict: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            moved[str(key)] = value.to(device=device)
        else:
            moved[str(key)] = copy.deepcopy(value)
    return moved


def _move_batch_to_device(batch: list[torch.Tensor], device: torch.device) -> list[torch.Tensor]:
    return [item.to(device=device) if isinstance(item, torch.Tensor) else item for item in batch]


def _validate_state_dict_compatible(
    live_state: dict[str, Any],
    trained_state: dict[str, Any],
) -> None:
    live_keys = set(live_state)
    trained_keys = set(trained_state)
    if live_keys != trained_keys:
        missing = sorted(live_keys - trained_keys)[:5]
        unexpected = sorted(trained_keys - live_keys)[:5]
        raise RuntimeError(
            "trained shadow state_dict keys do not match live model "
            f"missing={missing} unexpected={unexpected}"
        )
    for key, live_value in live_state.items():
        trained_value = trained_state[key]
        if isinstance(live_value, torch.Tensor) != isinstance(trained_value, torch.Tensor):
            raise RuntimeError(f"trained shadow state_dict type mismatch for {key}")
        if not isinstance(live_value, torch.Tensor):
            continue
        if tuple(live_value.shape) != tuple(trained_value.shape):
            raise RuntimeError(
                "trained shadow state_dict shape mismatch for "
                f"{key}: {tuple(trained_value.shape)} != {tuple(live_value.shape)}"
            )


def _prediction_from_artifacts(artifacts: dict[str, Any]) -> dict[str, Any]:
    prediction = dict(artifacts or {})
    prediction.setdefault("boxes", prediction.get("final_detection_boxes", []))
    prediction.setdefault("labels", prediction.get("final_detection_labels", []))
    prediction.setdefault("scores", prediction.get("final_detection_scores", []))
    entropy = _finite_float(prediction.get("output_entropy"))
    if entropy is None:
        entropy = _finite_float(prediction.get("logit_entropy"))
    if entropy is not None:
        prediction["output_entropy"] = entropy
    else:
        prediction.pop("output_entropy", None)
    return prediction


def _boundary_payload_from_artifacts(artifacts: dict[str, Any]) -> Any | None:
    for key in ("boundary_payload", "intermediate", "split_payload", "boundary"):
        value = artifacts.get(key)
        if value is not None:
            return value
    return None


def _output_entropy(prediction: dict[str, Any]) -> float | None:
    direct = _finite_float(prediction.get("output_entropy"))
    if direct is not None:
        return max(0.0, direct)
    scores = prediction.get("scores")
    if scores is None:
        return None
    try:
        tensor = torch.as_tensor(scores, dtype=torch.float32)
    except Exception:
        return None
    if tensor.numel() == 0:
        return None
    p = tensor.flatten()
    p = p[torch.isfinite(p)]
    if p.numel() == 0:
        return None
    p = p.clamp(0.0, 1.0)
    entropy = -(torch.xlogy(p, p) + torch.xlogy(1.0 - p, 1.0 - p))
    return _finite_float((entropy / math.log(2.0)).mean().item())


def _confidence(prediction: dict[str, Any]) -> float | None:
    direct = _finite_float(prediction.get("confidence"))
    if direct is not None:
        return max(0.0, min(1.0, direct))
    scores = prediction.get("scores")
    if scores is None:
        return None
    try:
        values = torch.as_tensor(scores, dtype=torch.float32).flatten()
    except Exception:
        return None
    if values.numel() == 0:
        return None
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return None
    return float(finite.clamp(0.0, 1.0).mean().item())


def _predictions_empty(prediction: dict[str, Any]) -> bool:
    boxes = prediction.get("boxes")
    if boxes is None:
        return True
    if isinstance(boxes, torch.Tensor):
        return boxes.numel() == 0 or (boxes.ndim > 0 and int(boxes.shape[0]) == 0)
    try:
        return len(list(boxes)) == 0
    except TypeError:
        return False


def _percentile(values: list[float], percentile: float) -> float | None:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    if len(finite) == 1:
        return float(finite[0])
    rank = max(0.0, min(100.0, float(percentile))) / 100.0 * (len(finite) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(finite[lower])
    fraction = rank - lower
    return float(finite[lower] * (1.0 - fraction) + finite[upper] * fraction)


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _extract_differentiable_logits(model: object, outputs: Any) -> tuple[torch.Tensor | None, str]:
    if isinstance(outputs, dict):
        explicit_mode = _valid_tta_logit_mode(outputs.get("_tta_logit_mode"))
        for key, mode in (
            ("pred_logits", "softmax_bg_last"),
            ("logits", "softmax_bg_last"),
            ("cls_logits", "sigmoid"),
            ("dense_logits", "softmax_bg_last"),
        ):
            value = outputs.get(key)
            if isinstance(value, torch.Tensor):
                if key == "pred_logits":
                    if explicit_mode is not None:
                        return value, explicit_mode
                    if _looks_like_rfdetr_tta_model(model):
                        return value, "sigmoid_bg_last"
                return value, mode
    if hasattr(outputs, "logits") and isinstance(outputs.logits, torch.Tensor):
        return outputs.logits, "softmax_bg_last"
    try:
        from model_management.split_model_adapters import _extract_runtime_logits

        return _extract_runtime_logits(model, outputs)
    except Exception:
        return None, "sigmoid"


def _valid_tta_logit_mode(value: object) -> str | None:
    mode = str(value or "").strip().lower()
    return mode if mode in {"sigmoid", "sigmoid_bg_last", "softmax", "softmax_bg_last"} else None


def _looks_like_rfdetr_tta_model(model: object) -> bool:
    name = type(model).__name__.lower()
    if "rfdetr" in name or "rf_detr" in name:
        return True
    if getattr(model, "rfdetr", None) is None:
        return False
    return callable(getattr(model, "_prepare_batch", None))


def _background_is_last(mode: object) -> bool:
    return str(mode or "").strip().lower().endswith("_bg_last")


def _align_horizontally_flipped_logits(
    logits: torch.Tensor,
    outputs: Any,
) -> torch.Tensor | None:
    if not isinstance(logits, torch.Tensor):
        return None
    if logits.ndim == 2:
        # Image-level logits have no spatial axis to realign.
        return logits
    if not isinstance(outputs, dict):
        return None
    flip_dim = outputs.get("_tta_horizontal_flip_dim")
    try:
        parsed_dim = int(flip_dim)
    except (TypeError, ValueError):
        return None
    if parsed_dim < 0:
        parsed_dim += logits.ndim
    if parsed_dim <= 0 or parsed_dim >= logits.ndim:
        return None
    return torch.flip(logits, dims=(parsed_dim,))


def _logit_rows(logits: torch.Tensor) -> torch.Tensor | None:
    if not isinstance(logits, torch.Tensor) or logits.numel() == 0:
        return None
    work = logits.float()
    if work.ndim == 1:
        return work.reshape(1, -1)
    if work.ndim == 2:
        return work
    if work.ndim == 3:
        if work.shape[-1] > 1 and work.shape[-1] <= 512:
            return work.reshape(-1, work.shape[-1])
        if work.shape[1] > 1 and work.shape[1] <= 512 and work.shape[2] > work.shape[1]:
            return work.permute(0, 2, 1).reshape(-1, work.shape[1])
        return work.reshape(-1, work.shape[-1])
    if work.ndim == 4:
        if work.shape[-1] > 1 and work.shape[-1] <= 512:
            return work.reshape(-1, work.shape[-1])
        if work.shape[1] > 1 and work.shape[1] <= 512:
            return work.permute(0, 2, 3, 1).reshape(-1, work.shape[1])
    return None


def _parameter_grad_state(model: torch.nn.Module) -> dict[str, bool]:
    return {name: bool(param.requires_grad) for name, param in model.named_parameters()}


def _batch_norm_tracking_state(model: torch.nn.Module) -> dict[str, bool]:
    return {
        name: bool(module.track_running_stats)
        for name, module in model.named_modules()
        if isinstance(
            module,
            (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d),
        )
    }


def _set_batch_norm_tracking(model: torch.nn.Module, *, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(
            module,
            (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d),
        ):
            module.track_running_stats = bool(enabled)


def _restore_batch_norm_tracking(model: torch.nn.Module, state: dict[str, bool]) -> None:
    for name, module in model.named_modules():
        if isinstance(
            module,
            (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d),
        ):
            module.track_running_stats = bool(state.get(name, module.track_running_stats))


def _module_training_state(model: torch.nn.Module) -> dict[str, bool]:
    return {name: bool(module.training) for name, module in model.named_modules()}


def _restore_module_training_state(
    model: torch.nn.Module,
    state: dict[str, bool],
    default_mode: bool,
) -> None:
    if hasattr(model, "train"):
        model.train(bool(default_mode))
    for name, module in model.named_modules():
        module.training = bool(state.get(name, default_mode))


def _restore_parameter_grad_state(model: torch.nn.Module, state: dict[str, bool]) -> None:
    for name, param in model.named_parameters():
        param.requires_grad_(state.get(name, param.requires_grad))


def _clear_gradients(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.grad = None


def _enable_norm_affine_parameters(model: torch.nn.Module) -> int:
    norm_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.GroupNorm,
        torch.nn.LayerNorm,
    )
    selected = 0
    for module in model.modules():
        if not isinstance(module, norm_types):
            continue
        for name in ("weight", "bias"):
            param = getattr(module, name, None)
            if isinstance(param, torch.nn.Parameter):
                param.requires_grad_(True)
                selected += int(param.numel())
    return selected


def _model_device(model: object) -> torch.device:
    if isinstance(model, torch.nn.Module):
        for param in model.parameters():
            return param.device
        for buffer in model.buffers():
            return buffer.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _next_surgeon_version(current: str) -> str:
    value = str(current or "0")
    if value.startswith("surgeon_"):
        try:
            return f"surgeon_{int(value.split('_', 1)[1]) + 1}"
        except (TypeError, ValueError):
            return "surgeon_1"
    return "surgeon_1"
