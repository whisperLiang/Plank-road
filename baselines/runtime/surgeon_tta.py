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
_PROBABILITY_EPS = 1.0e-6


@dataclass(frozen=True)
class _BufferedSample:
    frame_id: int
    frame: np.ndarray
    artifacts: dict[str, Any]
    latency_ms: float | None


@dataclass(frozen=True)
class TTALogitView:
    """Model-independent, row-major view of differentiable detector logits."""

    rows: torch.Tensor
    probabilities: torch.Tensor
    entropy: torch.Tensor
    foreground_mask: torch.Tensor
    mode: str


@dataclass(frozen=True)
class FrozenTTAReference:
    """Frozen row selection and probabilities captured before local adaptation."""

    probabilities: torch.Tensor
    selected_indices: torch.Tensor
    mode: str
    initial_stats: dict[str, Any]


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
    applied_epoch: int
    guard_stats: dict[str, Any]
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
        max_selected_logit_count: int = 256,
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
        self.max_selected_logit_count = max(
            self.min_selected_logit_count,
            int(max_selected_logit_count),
        )

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

        engine = getattr(model, "yolo", None) or getattr(model, "rtdetr", None)
        if getattr(engine, "model", None) is not None:
            return self._forward_ultralytics(model, images)

        raise _TTASkip("unsupported_tta_output_semantics")

    def logit_view(self, outputs: Any) -> TTALogitView:
        logits, mode = _extract_differentiable_logits(self.model, outputs)
        mode = _valid_tta_logit_mode(mode)
        if logits is None or not isinstance(logits, torch.Tensor) or mode is None:
            raise _TTASkip("unsupported_tta_output_semantics")
        rows = _logit_rows(logits)
        if rows is None or rows.numel() == 0:
            raise _TTASkip("unsupported_tta_output_semantics")
        if mode.startswith("sigmoid"):
            if _background_is_last(mode) and rows.shape[-1] > 1:
                rows = rows[:, :-1]
            if rows.shape[-1] < 1:
                raise _TTASkip("unsupported_tta_output_semantics")
            probs = torch.sigmoid(rows).clamp(_PROBABILITY_EPS, 1.0 - _PROBABILITY_EPS)
            p = probs.max(dim=-1).values
            foreground_mask = p.detach() >= _SIGMOID_FOREGROUND_PROBABILITY_FLOOR
            entropy = -((p * torch.log(p)) + ((1.0 - p) * torch.log(1.0 - p)))
            entropy = entropy / math.log(2.0)
        else:
            if rows.shape[-1] <= 1:
                raise _TTASkip("unsupported_tta_output_semantics")
            probs = torch.softmax(rows, dim=-1).clamp_min(_PROBABILITY_EPS)
            entropy = -(probs * torch.log(probs)).sum(dim=-1)
            entropy = entropy / max(math.log(max(2, int(rows.shape[-1]))), 1.0e-8)
            if _background_is_last(mode):
                foreground_mask = probs.detach().argmax(dim=-1) != int(rows.shape[-1] - 1)
            else:
                foreground_mask = (
                    probs.detach().max(dim=-1).values
                    >= _SIGMOID_FOREGROUND_PROBABILITY_FLOOR
                )

        return TTALogitView(
            rows=rows,
            probabilities=probs,
            entropy=entropy,
            foreground_mask=foreground_mask,
            mode=mode,
        )

    def capture_reference(
        self,
        outputs: Any,
        *,
        require_selection: bool = True,
    ) -> FrozenTTAReference:
        return self.capture_references(
            [outputs],
            require_selection=require_selection,
        )[0]

    def capture_references(
        self,
        outputs_batches: list[Any],
        *,
        require_selection: bool = True,
    ) -> list[FrozenTTAReference]:
        if not outputs_batches:
            raise _TTASkip("empty_batch")
        views = [self.logit_view(outputs) for outputs in outputs_batches]
        foreground_indices_by_view = [
            torch.nonzero(view.foreground_mask, as_tuple=False).flatten()
            for view in views
        ]
        foreground_entropy_by_view = [
            view.entropy.index_select(0, foreground_indices)
            for view, foreground_indices in zip(views, foreground_indices_by_view)
        ]
        foreground_entropy = torch.cat(foreground_entropy_by_view, dim=0)
        if require_selection and foreground_entropy.numel() == 0:
            raise _TTASkip("no_foreground_logits")

        strict_mask = foreground_entropy <= self.entropy_margin_ratio
        max_entropy_mask = foreground_entropy <= self.max_entropy_margin_ratio
        strict_selected_count = int(strict_mask.sum().item())
        max_entropy_candidate_count = int(max_entropy_mask.sum().item())
        adaptive_entropy_gate_used = False
        selected_positions = foreground_entropy.new_empty((0,), dtype=torch.long)
        if require_selection:
            if self.entropy_margin_ratio <= 0.0:
                candidate_local_indices = torch.arange(
                    foreground_entropy.numel(),
                    device=foreground_entropy.device,
                )
            elif strict_selected_count >= self.min_selected_logit_count:
                candidate_local_indices = torch.nonzero(strict_mask, as_tuple=False).flatten()
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
                candidate_local_indices = torch.nonzero(
                    max_entropy_mask,
                    as_tuple=False,
                ).flatten()
                adaptive_entropy_gate_used = True
            elif strict_selected_count > 0:
                candidate_local_indices = torch.nonzero(strict_mask, as_tuple=False).flatten()
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
            selection_count = min(
                int(candidate_local_indices.numel()),
                self.max_selected_logit_count,
            )
            if adaptive_entropy_gate_used:
                selection_count = min(selection_count, self.min_selected_logit_count)
            candidate_entropy = foreground_entropy.index_select(
                0,
                candidate_local_indices,
            )
            lowest = torch.topk(
                candidate_entropy,
                k=selection_count,
                largest=False,
                sorted=True,
            ).indices
            selected_positions = candidate_local_indices.index_select(0, lowest)

        references: list[FrozenTTAReference] = []
        foreground_offset = 0
        for view, foreground_indices, view_foreground_entropy in zip(
            views,
            foreground_indices_by_view,
            foreground_entropy_by_view,
        ):
            view_foreground_count = int(view_foreground_entropy.numel())
            in_view = (
                (selected_positions >= foreground_offset)
                & (selected_positions < foreground_offset + view_foreground_count)
            )
            selected_local_positions = (
                selected_positions[in_view] - foreground_offset
            )
            selected_indices = foreground_indices.index_select(
                0,
                selected_local_positions,
            )
            selected_entropy = view.entropy.index_select(0, selected_indices)
            initial_stats = {
                "logit_count": int(view.rows.shape[0]),
                "foreground_logit_count": view_foreground_count,
                "strict_selected_logit_count": int(
                    (view_foreground_entropy <= self.entropy_margin_ratio).sum().item()
                ),
                "max_entropy_candidate_count": int(
                    (view_foreground_entropy <= self.max_entropy_margin_ratio)
                    .sum()
                    .item()
                ),
                "selected_logit_count": int(selected_indices.numel()),
                "entropy": float(
                    view_foreground_entropy.detach().mean().item()
                    if view_foreground_entropy.numel()
                    else 0.0
                ),
                "effective_entropy_threshold": float(
                    selected_entropy.detach().max().item()
                    if selected_entropy.numel()
                    else 0.0
                ),
                "entropy_margin_ratio": self.entropy_margin_ratio,
                "max_entropy_margin_ratio": self.max_entropy_margin_ratio,
                "adaptive_entropy_gate": self.adaptive_entropy_gate,
                "adaptive_entropy_gate_used": adaptive_entropy_gate_used,
                "required_selected_logit_count": self.min_selected_logit_count,
                "max_selected_logit_count": self.max_selected_logit_count,
            }
            references.append(
                FrozenTTAReference(
                    probabilities=view.probabilities.detach().clone(),
                    selected_indices=selected_indices.detach().clone(),
                    mode=view.mode,
                    initial_stats=initial_stats,
                )
            )
            foreground_offset += view_foreground_count
        return references

    def anchored_loss(
        self,
        outputs: Any,
        reference: FrozenTTAReference,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        view = self._validated_reference_view(outputs, reference)
        selected_entropy = view.entropy.index_select(0, reference.selected_indices)
        entropy_loss = (
            selected_entropy.mean()
            if selected_entropy.numel()
            else view.entropy.sum() * 0.0
        )
        reference_kl = _normalized_reference_kl(
            reference.probabilities,
            view.probabilities,
            mode=view.mode,
        )
        stats = self._stats_for_view(view, reference.selected_indices)
        stats["reference_kl"] = float(reference_kl.detach().item())
        stats["frozen_selection"] = True
        stats["adaptive_entropy_gate_used"] = bool(
            reference.initial_stats.get("adaptive_entropy_gate_used", False)
        )
        return entropy_loss, reference_kl, stats

    def guard_stats(
        self,
        outputs: Any,
        reference: FrozenTTAReference,
    ) -> dict[str, Any]:
        view = self._validated_reference_view(outputs, reference)
        reference_kl = _normalized_reference_kl(
            reference.probabilities,
            view.probabilities,
            mode=view.mode,
        )
        current_foreground_count = int(view.foreground_mask.sum().item())
        reference_foreground_count = int(
            reference.initial_stats.get("foreground_logit_count", 0)
        )
        logit_count = max(1, int(view.rows.shape[0]))
        return {
            "logit_count": int(view.rows.shape[0]),
            "reference_foreground_logit_count": reference_foreground_count,
            "foreground_logit_count": current_foreground_count,
            "foreground_growth_ratio": float(
                current_foreground_count / max(1, reference_foreground_count)
            ),
            "foreground_fraction_increase": float(
                (current_foreground_count - reference_foreground_count) / logit_count
            ),
            "reference_kl": float(reference_kl.detach().item()),
            "logit_mode": view.mode,
        }

    def entropy_loss(self, outputs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        reference = self.capture_reference(outputs)
        entropy_loss, _reference_kl, stats = self.anchored_loss(outputs, reference)
        return entropy_loss, stats

    def _validated_reference_view(
        self,
        outputs: Any,
        reference: FrozenTTAReference,
    ) -> TTALogitView:
        view = self.logit_view(outputs)
        if (
            view.mode != reference.mode
            or tuple(view.probabilities.shape) != tuple(reference.probabilities.shape)
        ):
            raise _TTASkip(
                "unsupported_tta_output_semantics",
                reference_mode=reference.mode,
                current_mode=view.mode,
                reference_shape=list(reference.probabilities.shape),
                current_shape=list(view.probabilities.shape),
            )
        return view

    def _stats_for_view(
        self,
        view: TTALogitView,
        selected_indices: torch.Tensor,
    ) -> dict[str, Any]:
        foreground_entropy = view.entropy[view.foreground_mask]
        strict_selected_count = int(
            (foreground_entropy <= self.entropy_margin_ratio).sum().item()
        )
        max_entropy_candidate_count = int(
            (foreground_entropy <= self.max_entropy_margin_ratio).sum().item()
        )
        selected_entropy = view.entropy.index_select(0, selected_indices)
        return {
            "logit_count": int(view.rows.shape[0]),
            "foreground_logit_count": int(foreground_entropy.numel()),
            "strict_selected_logit_count": strict_selected_count,
            "max_entropy_candidate_count": max_entropy_candidate_count,
            "selected_logit_count": int(selected_indices.numel()),
            "entropy": float(
                foreground_entropy.detach().mean().item()
                if foreground_entropy.numel()
                else 0.0
            ),
            "effective_entropy_threshold": float(
                selected_entropy.detach().max().item()
                if selected_entropy.numel()
                else 0.0
            ),
            "entropy_margin_ratio": self.entropy_margin_ratio,
            "max_entropy_margin_ratio": self.max_entropy_margin_ratio,
            "adaptive_entropy_gate": self.adaptive_entropy_gate,
            "adaptive_entropy_gate_used": False,
            "required_selected_logit_count": self.min_selected_logit_count,
            "max_selected_logit_count": self.max_selected_logit_count,
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
            configured_train_sample_count = min(16, self.training_frame_count)
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
        self.max_selected_logit_count = max(
            self.min_selected_logit_count,
            int(getattr(self.method_cfg, "max_selected_logit_count", 256)),
        )
        self.min_loss_improvement = max(
            0.0,
            float(getattr(self.method_cfg, "min_loss_improvement", 1.0e-4)),
        )
        self.consistency_weight = max(
            0.0,
            float(getattr(self.method_cfg, "consistency_weight", 0.0)),
        )
        self.reference_consistency_weight = max(
            0.0,
            float(getattr(self.method_cfg, "reference_consistency_weight", 0.05)),
        )
        self.guard_sample_count = max(
            0,
            min(
                int(getattr(self.method_cfg, "guard_sample_count", 8)),
                max(0, self.training_frame_count - self.train_sample_count),
            ),
        )
        self.max_foreground_growth_ratio = max(
            1.0,
            float(getattr(self.method_cfg, "max_foreground_growth_ratio", 2.0)),
        )
        self.max_foreground_fraction_increase = max(
            0.0,
            float(getattr(self.method_cfg, "max_foreground_fraction_increase", 0.02)),
        )
        self.max_reference_kl = max(
            0.0,
            float(getattr(self.method_cfg, "max_reference_kl", 0.10)),
        )
        self.max_relative_param_delta = max(
            0.0,
            float(getattr(self.method_cfg, "max_relative_param_delta", 0.02)),
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
        configured_mini_batch_size = getattr(
            self.method_cfg,
            "mini_batch_size",
            None,
        )
        if configured_mini_batch_size is None:
            configured_mini_batch_size = getattr(self.training_cfg, "batch_size", 32)
        self.batch_size = max(1, int(configured_mini_batch_size))
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
            "train_sample_count={} guard_sample_count={} num_epoch={} "
            "batch_size={} quality_mode={} "
            "require_drift={} min_selected_logits={} entropy_margin={} "
            "adaptive_entropy_gate={} max_entropy_margin={} max_selected_logits={}",
            self.training_frame_count,
            self.train_sample_count,
            self.guard_sample_count,
            self.num_epoch,
            self.batch_size,
            self.quality_mode,
            self.require_drift,
            self.min_selected_logit_count,
            self.entropy_margin_ratio,
            self.adaptive_entropy_gate,
            self.max_entropy_margin_ratio,
            self.max_selected_logit_count,
        )
        self.metrics.record(
            "surgeon_tta_config",
            training_frame_count=int(self.training_frame_count),
            train_sample_count=int(self.train_sample_count),
            guard_sample_count=int(self.guard_sample_count),
            num_epoch=int(self.num_epoch),
            mini_batch_size=int(self.batch_size),
            trainable_scope=str(self.trainable_scope),
            consistency_weight=float(self.consistency_weight),
            reference_consistency_weight=float(self.reference_consistency_weight),
            min_loss_improvement=float(self.min_loss_improvement),
            min_selected_logit_count=int(self.min_selected_logit_count),
            max_selected_logit_count=int(self.max_selected_logit_count),
            entropy_margin_ratio=float(self.entropy_margin_ratio),
            adaptive_entropy_gate=bool(self.adaptive_entropy_gate),
            max_entropy_margin_ratio=float(self.max_entropy_margin_ratio),
            max_foreground_growth_ratio=float(self.max_foreground_growth_ratio),
            max_foreground_fraction_increase=float(
                self.max_foreground_fraction_increase
            ),
            max_reference_kl=float(self.max_reference_kl),
            max_relative_param_delta=float(self.max_relative_param_delta),
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
            buffered_samples = list(self._buffer)
            selected = buffered_samples[-int(self.train_sample_count) :]
            guard_pool = buffered_samples[: -int(self.train_sample_count)]
            guard_samples = _evenly_spaced_samples(
                guard_pool,
                self.guard_sample_count,
            )
            buffered_count = len(self._buffer)
            # Consume this trigger window. Samples observed while the shadow
            # model trains may form a later window, but a successful apply
            # discards them because they came from the old live model.
            self._buffer.clear()
            logger.info(
                "[SURGEON] local TTA triggered: low_quality={} "
                "training_frame_count={} train_sample_count={} guard_sample_count={} "
                "mini_batch_size={} trigger_frame={}",
                buffered_count,
                self.training_frame_count,
                len(selected),
                len(guard_samples),
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
                guard_sample_count=len(guard_samples),
                drift_detected=bool(drift.drift_detected),
                drift_score=float(drift.drift_score),
            )
            self._running_thread = threading.Thread(
                target=self._run_tta_task,
                args=(selected, guard_samples, int(frame_index)),
                name="pure-edge-surgeon-tta",
                daemon=True,
            )
            self._running_thread.start()

    def close(self, *, timeout: float = 10.0) -> None:
        self._closed.set()
        if self.wait_for_idle(timeout=max(0.0, float(timeout))):
            self.try_apply_pending_update()
        else:
            logger.warning(
                "[SURGEON] close timed out while shadow training was still running"
            )

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

    def _run_tta_task(
        self,
        samples: list[_BufferedSample],
        guard_samples: list[_BufferedSample],
        trigger_frame_id: int,
    ) -> None:
        started = time.perf_counter()
        self.metrics.record(
            "surgeon_tta_started",
            frame_id=int(trigger_frame_id),
            low_quality_sample_count=len(samples),
            batch_size=len(samples),
            guard_sample_count=len(guard_samples),
            num_epoch=self.num_epoch,
        )
        try:
            update = self._execute_tta(
                samples,
                guard_samples,
                trigger_frame_id,
                started,
            )
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
                    trained_epochs=int(update.num_epoch),
                    applied_epoch=int(update.applied_epoch),
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
                    **_prefixed_gate_stats(update.guard_stats, prefix="guard_"),
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
        guard_samples: list[_BufferedSample],
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
        if self._closed.is_set():
            raise _TTASkip("shutdown")
        self.metrics.record(
            "surgeon_tta_shadow_snapshot_done",
            frame_id=int(trigger_frame_id),
            snapshot_lock_ms=float(snapshot["snapshot_lock_ms"]),
            model_version_before=str(snapshot["model_version_before"]),
            model_class=str(snapshot["model_class"]),
            trainable_model_class=str(snapshot["trainable_model_class"]),
        )

        shadow_model = self._build_shadow_training_model(detector, snapshot)
        if self._closed.is_set():
            raise _TTASkip("shutdown")
        adapter = TTADetectionAdapter(
            detector,
            model_override=shadow_model,
            entropy_margin_ratio=self.entropy_margin_ratio,
            adaptive_entropy_gate=self.adaptive_entropy_gate,
            max_entropy_margin_ratio=self.max_entropy_margin_ratio,
            min_selected_logit_count=self.min_selected_logit_count,
            max_selected_logit_count=self.max_selected_logit_count,
        )
        batch = adapter.build_batch([sample.frame for sample in samples])
        guard_batch = adapter.build_batch(
            [sample.frame for sample in guard_samples]
        ) if guard_samples else list(batch)
        train_result = self._train_shadow_model(
            adapter=adapter,
            batch=batch,
            guard_batch=guard_batch,
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
            applied_epoch=int(train_result["applied_epoch"]),
            guard_stats=dict(train_result["guard_stats"]),
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
            trained_epochs=int(update.num_epoch),
            applied_epoch=int(update.applied_epoch),
            loss=float(update.loss),
            initial_loss=float(update.initial_loss),
            initial_selected_logit_count=int(update.initial_selected_logit_count),
            selected_logit_count=int(update.selected_logit_count),
            **_prefixed_gate_stats(
                update.initial_gate_stats,
                prefix="initial_",
            ),
            **update.gate_stats,
            **_prefixed_gate_stats(update.guard_stats, prefix="guard_"),
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
        guard_batch: list[torch.Tensor],
        trigger_frame_id: int,
        model_version_before: str,
    ) -> dict[str, Any]:
        trainable_model = adapter.trainable_model
        training_device = _model_device(trainable_model)
        batch = list(batch)
        guard_batch = list(guard_batch)
        total_sample_count = len(batch)
        mini_batch_size = max(1, int(self.batch_size))
        mini_batches = [
            batch[index : index + mini_batch_size]
            for index in range(0, total_sample_count, mini_batch_size)
        ]
        guard_mini_batches = [
            guard_batch[index : index + mini_batch_size]
            for index in range(0, len(guard_batch), mini_batch_size)
        ]
        if not mini_batches or not guard_mini_batches:
            raise _TTASkip("empty_batch")
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
            max_selected_logit_count=int(self.max_selected_logit_count),
            reference_consistency_weight=float(self.reference_consistency_weight),
            guard_sample_count=int(len(guard_batch)),
            max_foreground_growth_ratio=float(self.max_foreground_growth_ratio),
            max_foreground_fraction_increase=float(
                self.max_foreground_fraction_increase
            ),
            max_reference_kl=float(self.max_reference_kl),
            max_relative_param_delta=float(self.max_relative_param_delta),
        )
        logger.info(
            "[SURGEON] shadow training started: samples={} "
            "mini_batch_size={} epochs={}",
            int(total_sample_count),
            int(mini_batch_size),
            int(self.num_epoch),
        )
        try:
            if self._closed.is_set():
                raise _TTASkip("shutdown")
            self._select_trainable_parameters(trainable_model)
            trainable_named_params = [
                (name, param)
                for name, param in trainable_model.named_parameters()
                if param.requires_grad
            ]
            trainable_params = [param for _, param in trainable_named_params]
            trainable_param_count = sum(int(param.numel()) for param in trainable_params)
            if not trainable_params:
                raise _TTASkip("no_trainable_parameters")
            parameter_reference = {
                name: param.detach().clone()
                for name, param in trainable_named_params
            }
            optimizer = self._make_optimizer(trainable_params)
            _set_batch_norm_tracking(trainable_model, enabled=False)
            if hasattr(trainable_model, "train"):
                trainable_model.train(True)
            optimizer.zero_grad(set_to_none=True)
            train_references = self._capture_references(
                adapter=adapter,
                mini_batches=mini_batches,
                training_device=training_device,
                require_selection=True,
            )
            guard_references = self._capture_references(
                adapter=adapter,
                mini_batches=guard_mini_batches,
                training_device=training_device,
                require_selection=False,
            )
            initial_objective = self._evaluate_fixed_objective(
                adapter=adapter,
                mini_batches=mini_batches,
                references=train_references,
                training_device=training_device,
            )
            initial_guard_stats = self._evaluate_guard_set(
                adapter=adapter,
                mini_batches=guard_mini_batches,
                references=guard_references,
                training_device=training_device,
            )
            self.metrics.record(
                "surgeon_tta_guard_reference",
                frame_id=int(trigger_frame_id),
                **initial_guard_stats,
                model_version_before=str(model_version_before),
            )
            initial_loss = float(initial_objective["loss"])
            total_fixed_selected_count = max(
                1,
                int(initial_objective["selected_logit_count"]),
            )
            total_reference_logit_count = max(
                1,
                int(initial_objective["logit_count"]),
            )
            best_epoch: int | None = None
            best_loss = float("inf")
            best_parameter_state: dict[str, torch.Tensor] | None = None
            last_rejection_reasons: list[str] = []
            last_training_loss = initial_loss
            for epoch_index in range(self.num_epoch):
                if self._closed.is_set():
                    raise _TTASkip("shutdown", trained_epochs=epoch_index)
                epoch_started = time.perf_counter()
                epoch = epoch_index + 1
                epoch_loss_values: list[float] = []
                epoch_legacy_consistency_values: list[float] = []
                epoch_consistency_batch_count = 0
                projection_applied = False
                for mini_batch, reference in zip(mini_batches, train_references):
                    device_batch = _move_batch_to_device(mini_batch, training_device)
                    optimizer.zero_grad(set_to_none=True)
                    outputs = adapter.forward_tta_outputs(device_batch)
                    entropy_loss, reference_kl, loss_stats = adapter.anchored_loss(
                        outputs,
                        reference,
                    )
                    entropy_weight = (
                        int(loss_stats["selected_logit_count"])
                        / total_fixed_selected_count
                    )
                    reference_weight = (
                        int(loss_stats["logit_count"])
                        / total_reference_logit_count
                    )
                    weighted_reference_kl = (
                        self.reference_consistency_weight
                        * reference_weight
                        * reference_kl
                    )
                    loss = entropy_weight * entropy_loss + weighted_reference_kl
                    legacy_consistency_value = 0.0
                    if self.consistency_weight > 0.0:
                        augmented = adapter.forward_tta_outputs(device_batch, augment=True)
                        consistency = adapter.consistency_loss(outputs, augmented)
                        if consistency is not None:
                            epoch_consistency_batch_count += 1
                            legacy_consistency_value = float(consistency.detach().item())
                            weighted_consistency = self.consistency_weight * consistency
                            loss = loss + weighted_consistency
                    if not bool(torch.isfinite(loss).item()):
                        raise _TTASkip(
                            "no_safe_tta_candidate",
                            trained_epochs=epoch - 1,
                            rejection_reasons=["nonfinite_training_loss"],
                        )
                    loss.backward()
                    optimizer.step()
                    projection_stats = _project_named_parameters(
                        trainable_named_params,
                        parameter_reference,
                        self.max_relative_param_delta,
                    )
                    if not bool(projection_stats["finite"]):
                        raise _TTASkip(
                            "no_safe_tta_candidate",
                            trained_epochs=epoch,
                            rejection_reasons=["nonfinite_parameter_update"],
                        )
                    projection_applied = projection_applied or bool(
                        projection_stats["projected"]
                    )
                    epoch_loss_values.append(float(loss.detach().item()))
                    epoch_legacy_consistency_values.append(legacy_consistency_value)

                candidate_objective = self._evaluate_fixed_objective(
                    adapter=adapter,
                    mini_batches=mini_batches,
                    references=train_references,
                    training_device=training_device,
                )
                guard_stats = self._evaluate_guard_set(
                    adapter=adapter,
                    mini_batches=guard_mini_batches,
                    references=guard_references,
                    training_device=training_device,
                )
                parameter_delta = _relative_named_parameter_delta(
                    trainable_named_params,
                    parameter_reference,
                )
                rejection_reasons = self._candidate_rejection_reasons(
                    objective=candidate_objective,
                    guard_stats=guard_stats,
                    parameter_delta=parameter_delta,
                )
                candidate_safe = not rejection_reasons
                candidate_loss = float(candidate_objective["loss"])
                objective_improved = math.isfinite(candidate_loss) and (
                    candidate_loss < initial_loss - self.min_loss_improvement
                )
                if not objective_improved:
                    rejection_reasons = [*rejection_reasons, "objective_not_improved"]
                if candidate_safe and objective_improved and candidate_loss < best_loss:
                    best_epoch = epoch
                    best_loss = candidate_loss
                    best_parameter_state = {
                        name: param.detach().clone()
                        for name, param in trainable_named_params
                    }
                last_rejection_reasons = list(dict.fromkeys(rejection_reasons))
                last_training_loss = (
                    float(np.sum(epoch_loss_values))
                    if epoch_loss_values
                    else float("inf")
                )
                epoch_ms = (time.perf_counter() - epoch_started) * 1000.0
                logger.info(
                    "[SURGEON][Train] epoch={}/{} loss={:.6f} entropy={:.6f} "
                    "reference_kl={:.6f} fixed_logits={} guard_fg={}/{} "
                    "guard_growth={:.4f} param_delta={:.6f} safe={} best_epoch={} "
                    "model_version={} epoch_ms={:.3f}",
                    epoch,
                    self.num_epoch,
                    candidate_loss,
                    float(candidate_objective["entropy_loss"]),
                    float(candidate_objective["reference_kl"]),
                    int(candidate_objective["selected_logit_count"]),
                    int(guard_stats["foreground_logit_count"]),
                    int(guard_stats["reference_foreground_logit_count"]),
                    float(guard_stats["foreground_growth_ratio"]),
                    float(parameter_delta),
                    candidate_safe,
                    best_epoch,
                    model_version_before,
                    epoch_ms,
                )
                self.metrics.record(
                    "surgeon_tta_epoch",
                    frame_id=int(trigger_frame_id),
                    epoch=int(epoch),
                    total_epochs=int(self.num_epoch),
                    loss=candidate_loss,
                    training_loss=float(last_training_loss),
                    entropy_loss=float(candidate_objective["entropy_loss"]),
                    reference_kl=float(candidate_objective["reference_kl"]),
                    weighted_reference_kl=float(
                        self.reference_consistency_weight
                        * float(candidate_objective["reference_kl"])
                    ),
                    consistency_loss=float(
                        np.mean(epoch_legacy_consistency_values)
                        if epoch_legacy_consistency_values
                        else 0.0
                    ),
                    batch_size=int(total_sample_count),
                    mini_batch_size=int(mini_batch_size),
                    logit_count=int(candidate_objective["logit_count"]),
                    foreground_logit_count=int(
                        candidate_objective["foreground_logit_count"]
                    ),
                    strict_selected_logit_count=int(
                        candidate_objective["strict_selected_logit_count"]
                    ),
                    max_entropy_candidate_count=int(
                        candidate_objective["max_entropy_candidate_count"]
                    ),
                    selected_logit_count=int(
                        candidate_objective["selected_logit_count"]
                    ),
                    frozen_selection=True,
                    max_selected_logit_count=int(self.max_selected_logit_count),
                    effective_entropy_threshold=float(
                        candidate_objective["effective_entropy_threshold"]
                    ),
                    adaptive_entropy_gate=bool(self.adaptive_entropy_gate),
                    adaptive_entropy_gate_used=bool(
                        candidate_objective["adaptive_entropy_gate_used"]
                    ),
                    adaptive_entropy_gate_batch_count=int(
                        candidate_objective["adaptive_entropy_gate_batch_count"]
                    ),
                    entropy_margin_ratio=float(self.entropy_margin_ratio),
                    max_entropy_margin_ratio=float(self.max_entropy_margin_ratio),
                    required_selected_logit_count=int(
                        self.min_selected_logit_count
                    ),
                    consistency_batch_count=int(epoch_consistency_batch_count),
                    parameter_delta_ratio=float(parameter_delta),
                    parameter_projection_applied=bool(projection_applied),
                    candidate_safe=bool(candidate_safe),
                    objective_improved=bool(objective_improved),
                    rejection_reasons=list(dict.fromkeys(rejection_reasons)),
                    best_epoch=best_epoch,
                    **_prefixed_gate_stats(guard_stats, prefix="guard_"),
                    model_version=str(model_version_before),
                    epoch_ms=float(epoch_ms),
                )
            optimizer.zero_grad(set_to_none=True)
            if best_parameter_state is None or best_epoch is None:
                self.metrics.record(
                    "surgeon_tta_rejected",
                    frame_id=int(trigger_frame_id),
                    reason="no_safe_tta_candidate",
                    trained_epochs=int(self.num_epoch),
                    initial_loss=float(initial_loss),
                    required_improvement=float(self.min_loss_improvement),
                    rejection_reasons=last_rejection_reasons,
                    model_version_before=str(model_version_before),
                )
                raise _TTASkip(
                    "no_safe_tta_candidate",
                    trained_epochs=int(self.num_epoch),
                    initial_loss=float(initial_loss),
                    rejection_reasons=last_rejection_reasons,
                )
            _restore_named_parameters(trainable_named_params, best_parameter_state)
            final_objective = self._evaluate_fixed_objective(
                adapter=adapter,
                mini_batches=mini_batches,
                references=train_references,
                training_device=training_device,
            )
            final_loss = float(final_objective["loss"])
            final_guard_stats = self._evaluate_guard_set(
                adapter=adapter,
                mini_batches=guard_mini_batches,
                references=guard_references,
                training_device=training_device,
            )
            final_parameter_delta = _relative_named_parameter_delta(
                trainable_named_params,
                parameter_reference,
            )
            final_rejection_reasons = self._candidate_rejection_reasons(
                objective=final_objective,
                guard_stats=final_guard_stats,
                parameter_delta=final_parameter_delta,
            )
            if final_loss >= initial_loss - self.min_loss_improvement:
                final_rejection_reasons.append("objective_not_improved")
            final_rejection_reasons = list(dict.fromkeys(final_rejection_reasons))
            if final_rejection_reasons:
                self.metrics.record(
                    "surgeon_tta_rejected",
                    frame_id=int(trigger_frame_id),
                    reason="no_safe_tta_candidate",
                    trained_epochs=int(self.num_epoch),
                    applied_epoch=int(best_epoch),
                    initial_loss=float(initial_loss),
                    final_loss=float(final_loss),
                    rejection_reasons=final_rejection_reasons,
                    **_prefixed_gate_stats(final_guard_stats, prefix="guard_"),
                    parameter_delta_ratio=float(final_parameter_delta),
                    model_version_before=str(model_version_before),
                )
                raise _TTASkip(
                    "no_safe_tta_candidate",
                    trained_epochs=int(self.num_epoch),
                    applied_epoch=int(best_epoch),
                    rejection_reasons=final_rejection_reasons,
                )
            shadow_train_ms = (time.perf_counter() - train_started) * 1000.0
            self.metrics.record(
                "surgeon_tta_shadow_train_done",
                frame_id=int(trigger_frame_id),
                batch_size=int(total_sample_count),
                mini_batch_size=int(mini_batch_size),
                num_epoch=int(self.num_epoch),
                trained_epochs=int(self.num_epoch),
                applied_epoch=int(best_epoch),
                initial_loss=initial_loss,
                loss=float(final_loss),
                training_loss=float(last_training_loss),
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
                **_prefixed_gate_stats(final_guard_stats, prefix="guard_"),
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
                "guard_stats": dict(final_guard_stats),
                "applied_epoch": int(best_epoch),
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

    def _capture_references(
        self,
        *,
        adapter: TTADetectionAdapter,
        mini_batches: list[list[torch.Tensor]],
        training_device: torch.device,
        require_selection: bool,
    ) -> list[FrozenTTAReference]:
        outputs_batches: list[Any] = []
        with torch.no_grad():
            for mini_batch in mini_batches:
                device_batch = _move_batch_to_device(mini_batch, training_device)
                outputs_batches.append(adapter.forward_tta_outputs(device_batch))
            references = adapter.capture_references(
                outputs_batches,
                require_selection=require_selection,
            )
        if require_selection:
            self._require_reliable_logits(
                {
                    "selected_logit_count": sum(
                        int(reference.initial_stats["selected_logit_count"])
                        for reference in references
                    ),
                    "strict_selected_logit_count": sum(
                        int(reference.initial_stats["strict_selected_logit_count"])
                        for reference in references
                    ),
                    "max_entropy_candidate_count": sum(
                        int(reference.initial_stats["max_entropy_candidate_count"])
                        for reference in references
                    ),
                }
            )
        return references

    def _evaluate_fixed_objective(
        self,
        *,
        adapter: TTADetectionAdapter,
        mini_batches: list[list[torch.Tensor]],
        references: list[FrozenTTAReference],
        training_device: torch.device,
    ) -> dict[str, float | int | bool]:
        weighted_entropy_sum = 0.0
        weighted_reference_kl_sum = 0.0
        logit_count = 0
        foreground_logit_count = 0
        strict_selected_logit_count = 0
        max_entropy_candidate_count = 0
        selected_logit_count = 0
        effective_entropy_thresholds: list[float] = []
        adaptive_entropy_gate_batch_count = 0
        with torch.no_grad():
            for mini_batch, reference in zip(mini_batches, references):
                device_batch = _move_batch_to_device(mini_batch, training_device)
                outputs = adapter.forward_tta_outputs(device_batch)
                entropy_loss, reference_kl, loss_stats = adapter.anchored_loss(
                    outputs,
                    reference,
                )
                batch_logit_count = int(loss_stats.get("logit_count", 0))
                batch_selected_count = int(
                    loss_stats.get("selected_logit_count", 0)
                )
                weighted_entropy_sum += (
                    float(entropy_loss.detach().item()) * batch_selected_count
                )
                weighted_reference_kl_sum += (
                    float(reference_kl.detach().item()) * batch_logit_count
                )
                logit_count += batch_logit_count
                foreground_logit_count += int(
                    loss_stats.get("foreground_logit_count", 0)
                )
                strict_selected_logit_count += int(
                    loss_stats.get("strict_selected_logit_count", 0)
                )
                max_entropy_candidate_count += int(
                    loss_stats.get("max_entropy_candidate_count", 0)
                )
                selected_logit_count += batch_selected_count
                effective_threshold = _finite_float(
                    loss_stats.get("effective_entropy_threshold")
                )
                if effective_threshold is not None:
                    effective_entropy_thresholds.append(effective_threshold)
                if bool(loss_stats.get("adaptive_entropy_gate_used", False)):
                    adaptive_entropy_gate_batch_count += 1
        entropy_loss_value = (
            weighted_entropy_sum / selected_logit_count
            if selected_logit_count
            else float("inf")
        )
        reference_kl_value = (
            weighted_reference_kl_sum / logit_count
            if logit_count
            else float("inf")
        )
        return {
            "loss": float(
                entropy_loss_value
                + self.reference_consistency_weight * reference_kl_value
            ),
            "entropy_loss": float(entropy_loss_value),
            "reference_kl": float(reference_kl_value),
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
            "frozen_selection": True,
            "max_selected_logit_count": int(self.max_selected_logit_count),
        }

    def _evaluate_guard_set(
        self,
        *,
        adapter: TTADetectionAdapter,
        mini_batches: list[list[torch.Tensor]],
        references: list[FrozenTTAReference],
        training_device: torch.device,
    ) -> dict[str, Any]:
        logit_count = 0
        reference_foreground_count = 0
        foreground_count = 0
        weighted_reference_kl = 0.0
        modes: set[str] = set()
        with torch.no_grad():
            for mini_batch, reference in zip(mini_batches, references):
                device_batch = _move_batch_to_device(mini_batch, training_device)
                stats = adapter.guard_stats(
                    adapter.forward_tta_outputs(device_batch),
                    reference,
                )
                count = int(stats["logit_count"])
                logit_count += count
                reference_foreground_count += int(
                    stats["reference_foreground_logit_count"]
                )
                foreground_count += int(stats["foreground_logit_count"])
                weighted_reference_kl += float(stats["reference_kl"]) * count
                modes.add(str(stats["logit_mode"]))
        if reference_foreground_count == 0:
            growth_ratio = 1.0 if foreground_count == 0 else float("inf")
        else:
            growth_ratio = foreground_count / reference_foreground_count
        return {
            "sample_count": int(sum(len(batch) for batch in mini_batches)),
            "logit_count": int(logit_count),
            "reference_foreground_logit_count": int(reference_foreground_count),
            "foreground_logit_count": int(foreground_count),
            "foreground_growth_ratio": float(growth_ratio),
            "foreground_fraction_increase": float(
                (foreground_count - reference_foreground_count) / max(1, logit_count)
            ),
            "reference_kl": float(weighted_reference_kl / max(1, logit_count)),
            "logit_mode": ",".join(sorted(modes)),
        }

    def _candidate_rejection_reasons(
        self,
        *,
        objective: dict[str, Any],
        guard_stats: dict[str, Any],
        parameter_delta: float,
    ) -> list[str]:
        values = [
            float(objective.get("loss", float("nan"))),
            float(guard_stats.get("foreground_growth_ratio", float("nan"))),
            float(guard_stats.get("foreground_fraction_increase", float("nan"))),
            float(guard_stats.get("reference_kl", float("nan"))),
            float(parameter_delta),
        ]
        if not all(math.isfinite(value) for value in values):
            return ["nonfinite_candidate_metrics"]
        reasons: list[str] = []
        if float(guard_stats["foreground_growth_ratio"]) > self.max_foreground_growth_ratio:
            reasons.append("foreground_growth_exceeded")
        if (
            float(guard_stats["foreground_fraction_increase"])
            > self.max_foreground_fraction_increase
        ):
            reasons.append("foreground_fraction_increase_exceeded")
        if float(guard_stats["reference_kl"]) > self.max_reference_kl:
            reasons.append("reference_kl_exceeded")
        parameter_tolerance = max(
            1.0e-6,
            self.max_relative_param_delta * 1.0e-5,
        )
        if float(parameter_delta) > self.max_relative_param_delta + parameter_tolerance:
            reasons.append("parameter_delta_exceeded")
        return reasons

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
        "reference_kl": float(stats.get("reference_kl", 0.0)),
        "frozen_selection": bool(stats.get("frozen_selection", False)),
        "max_selected_logit_count": int(
            stats.get("max_selected_logit_count", 0)
        ),
    }


def _prefixed_gate_stats(
    stats: dict[str, Any],
    *,
    prefix: str,
) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in stats.items()}


def _evenly_spaced_samples(
    samples: list[_BufferedSample],
    count: int,
) -> list[_BufferedSample]:
    requested = max(0, int(count))
    if requested <= 0 or not samples:
        return []
    if requested >= len(samples):
        return list(samples)
    indices = np.linspace(0, len(samples) - 1, num=requested, dtype=int)
    return [samples[int(index)] for index in indices]


def _relative_named_parameter_delta(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    reference: dict[str, torch.Tensor],
) -> float:
    delta_squared = 0.0
    reference_squared = 0.0
    for name, parameter in named_parameters:
        current = parameter.detach()
        before = reference[name].to(device=current.device, dtype=current.dtype)
        delta_squared += float(torch.sum((current - before) ** 2).item())
        reference_squared += float(torch.sum(before**2).item())
    if not math.isfinite(delta_squared) or not math.isfinite(reference_squared):
        return float("inf")
    denominator = max(math.sqrt(reference_squared), 1.0e-12)
    return math.sqrt(max(0.0, delta_squared)) / denominator


def _project_named_parameters(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    reference: dict[str, torch.Tensor],
    max_relative_delta: float,
) -> dict[str, Any]:
    ratio_before = _relative_named_parameter_delta(named_parameters, reference)
    if not math.isfinite(ratio_before):
        return {
            "finite": False,
            "projected": False,
            "ratio_before": ratio_before,
            "ratio_after": ratio_before,
        }
    radius = max(0.0, float(max_relative_delta))
    projected = ratio_before > radius
    if projected:
        scale = 0.0 if ratio_before <= 0.0 else radius / ratio_before
        with torch.no_grad():
            for name, parameter in named_parameters:
                before = reference[name].to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                parameter.copy_(before + (parameter - before) * scale)
    ratio_after = _relative_named_parameter_delta(named_parameters, reference)
    return {
        "finite": math.isfinite(ratio_after),
        "projected": bool(projected),
        "ratio_before": float(ratio_before),
        "ratio_after": float(ratio_after),
    }


def _restore_named_parameters(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    state: dict[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        for name, parameter in named_parameters:
            parameter.copy_(
                state[name].to(device=parameter.device, dtype=parameter.dtype)
            )


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


def _normalized_reference_kl(
    reference_probabilities: torch.Tensor,
    current_probabilities: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    if tuple(reference_probabilities.shape) != tuple(current_probabilities.shape):
        raise _TTASkip("unsupported_tta_output_semantics")
    reference = reference_probabilities.detach().clamp(
        _PROBABILITY_EPS,
        1.0 - _PROBABILITY_EPS,
    )
    current = current_probabilities.clamp(
        _PROBABILITY_EPS,
        1.0 - _PROBABILITY_EPS,
    )
    if str(mode).startswith("sigmoid"):
        divergence = reference * torch.log(reference / current)
        divergence = divergence + (1.0 - reference) * torch.log(
            (1.0 - reference) / (1.0 - current)
        )
        return divergence.mean() / math.log(2.0)
    divergence = (reference * torch.log(reference / current)).sum(dim=-1)
    return divergence.mean() / max(
        math.log(max(2, int(reference.shape[-1]))),
        1.0e-8,
    )


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
                if explicit_mode is not None:
                    return value, explicit_mode
                if key == "pred_logits":
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
