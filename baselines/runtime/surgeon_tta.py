from __future__ import annotations

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
from model_management.activation_sparsity import apply_das_to_model, compute_tgi


@dataclass(frozen=True)
class _BufferedSample:
    frame_id: int
    frame: np.ndarray
    artifacts: dict[str, Any]
    latency_ms: float | None


class _TTASkip(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class TTADetectionAdapter:
    """Small differentiable-output adapter for local detector TTA."""

    def __init__(self, detector: object, *, entropy_margin_ratio: float = 0.4) -> None:
        self.detector = detector
        self.model = getattr(detector, "model", detector)
        self.trainable_model = unwrap_trainable_module(
            self.model,
            model_name=str(getattr(detector, "model_name", "")),
        )
        self.entropy_margin_ratio = max(0.0, min(1.0, float(entropy_margin_ratio)))

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
        model_type = type(model).__name__
        if model_type == "RFDETRDetectionModel" and hasattr(model, "_prepare_batch"):
            batch_tensor, _ = model._prepare_batch(images)
            outputs = model.rfdetr.model.model(batch_tensor)
            if isinstance(outputs, tuple):
                return {"pred_logits": outputs[1], "pred_boxes": outputs[0]}
            return outputs

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
        if mode == "softmax_bg_last" and rows.shape[-1] > 1:
            rows = rows[:, :-1]
        if rows.shape[-1] <= 1:
            raise _TTASkip("logits_unavailable")

        if str(mode).startswith("sigmoid"):
            probs = torch.sigmoid(rows)
            p = probs.max(dim=-1).values.clamp(1.0e-8, 1.0 - 1.0e-8)
            entropy = -((p * torch.log(p)) + ((1.0 - p) * torch.log(1.0 - p)))
            entropy = entropy / math.log(2.0)
        else:
            probs = torch.softmax(rows, dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1.0e-8))).sum(dim=-1)
            entropy = entropy / max(math.log(max(2, int(rows.shape[-1]))), 1.0e-8)

        selected = entropy
        if self.entropy_margin_ratio > 0.0:
            mask = entropy <= self.entropy_margin_ratio
            if bool(mask.any()):
                selected = entropy[mask]
            elif entropy.numel() > 0:
                selected = entropy.topk(k=1, largest=False).values
        loss = selected.mean()
        return loss, {
            "logit_count": int(rows.shape[0]),
            "selected_logit_count": int(selected.numel()),
            "entropy": float(entropy.detach().mean().item()),
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
        if str(mode_a).startswith("sigmoid"):
            probs_a = torch.sigmoid(logits_a.detach())
            probs_b = torch.sigmoid(logits_b)
        else:
            work_a = logits_a[..., :-1] if mode_a == "softmax_bg_last" else logits_a
            work_b = logits_b[..., :-1] if mode_b == "softmax_bg_last" else logits_b
            if tuple(work_a.shape) != tuple(work_b.shape) or work_a.shape[-1] <= 1:
                return None
            probs_a = torch.softmax(work_a.detach(), dim=-1)
            probs_b = torch.softmax(work_b, dim=-1)
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


class SurgeonLocalTTAUpdater:
    def __init__(self, config: object, metrics_writer: object) -> None:
        self.config = config
        self.metrics = metrics_writer
        baseline_cfg = getattr(config, "baseline", None)
        self.method_cfg = getattr(baseline_cfg, "pure_edge_local_updating", None)
        self.training_cfg = getattr(baseline_cfg, "training", None)
        self.das_cfg = getattr(config, "das", None)
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
        self._output_min_confidence = max(
            0.0,
            min(
                1.0,
                float(
                    getattr(
                        getattr(quality_cfg, "output_entropy", None),
                        "min_detection_confidence",
                        0.85,
                    )
                ),
            ),
        )
        self.drift_detector = WindowDriftDetector(
            window_size=int(getattr(drift_cfg, "window_size", 100)),
            min_window_size=int(getattr(drift_cfg, "min_window_size", 30)),
            low_quality_rate_threshold=float(getattr(drift_cfg, "low_quality_rate_threshold", 0.3)),
            persistence_windows=int(getattr(drift_cfg, "persistence_windows", 3)),
        )

        self.trigger_low_quality_samples = max(
            1,
            int(getattr(self.method_cfg, "trigger_low_quality_samples", 8)),
        )
        self.max_local_buffer_samples = max(
            1,
            int(getattr(self.method_cfg, "max_local_buffer_samples", 64)),
        )
        self.tta_steps = max(1, int(getattr(self.method_cfg, "tta_steps", 1)))
        self.consistency_weight = max(
            0.0,
            float(getattr(self.method_cfg, "consistency_weight", 0.01)),
        )
        self.entropy_margin_ratio = max(
            0.0,
            min(1.0, float(getattr(self.method_cfg, "entropy_margin_ratio", 0.4))),
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
        self._buffer: deque[_BufferedSample] = deque(maxlen=self.max_local_buffer_samples)
        self._lock = threading.Lock()
        self._running_thread: threading.Thread | None = None
        self._closed = threading.Event()
        self._das_trainer = None
        self._das_model_id: int | None = None

    def attach_edge(self, edge) -> None:
        self._edge = edge

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
            if len(self._buffer) < self.trigger_low_quality_samples:
                return
            if self._running_thread is not None and self._running_thread.is_alive():
                return
            selected = list(self._buffer)[: min(len(self._buffer), self.batch_size)]
            self.metrics.record(
                "surgeon_tta_triggered",
                frame_id=int(frame_index),
                low_quality_sample_count=len(self._buffer),
                batch_size=len(selected),
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
        frame_ids = {int(sample.frame_id) for sample in samples}
        self.metrics.record(
            "surgeon_tta_started",
            frame_id=int(trigger_frame_id),
            low_quality_sample_count=len(samples),
            batch_size=len(samples),
            tta_steps=self.tta_steps,
        )
        try:
            result = self._execute_tta(samples)
        except _TTASkip as exc:
            self.metrics.record(
                "surgeon_tta_skipped",
                frame_id=int(trigger_frame_id),
                reason=exc.reason,
                low_quality_sample_count=len(samples),
            )
        except Exception as exc:  # noqa: BLE001 - metrics must capture runtime failures.
            logger.warning("[PureEdgeSURGEON] local TTA failed: {}", exc)
            self.metrics.record(
                "surgeon_tta_failed",
                frame_id=int(trigger_frame_id),
                message=str(exc),
                low_quality_sample_count=len(samples),
            )
        else:
            duration_ms = (time.perf_counter() - started) * 1000.0
            self.metrics.record(
                "surgeon_tta_done",
                frame_id=int(trigger_frame_id),
                low_quality_sample_count=len(samples),
                batch_size=int(result["batch_size"]),
                tta_steps=self.tta_steps,
                loss=float(result["loss"]),
                duration_ms=duration_ms,
                das_enabled=bool(result["das_enabled"]),
                trainable_param_count=int(result["trainable_param_count"]),
                model_version_before=str(result["model_version_before"]),
                model_version_after=str(result["model_version_after"]),
            )
            self.metrics.record(
                "local_model_update_applied",
                frame_id=int(trigger_frame_id),
                model_version_before=str(result["model_version_before"]),
                model_version_after=str(result["model_version_after"]),
            )
        finally:
            with self._lock:
                self._buffer = deque(
                    (sample for sample in self._buffer if int(sample.frame_id) not in frame_ids),
                    maxlen=self.max_local_buffer_samples,
                )
                self._running_thread = None

    def _execute_tta(self, samples: list[_BufferedSample]) -> dict[str, Any]:
        if self._edge is None:
            raise _TTASkip("edge_unavailable")
        detector = getattr(self._edge, "small_object_detection", None)
        if detector is None:
            raise _TTASkip("detector_unavailable")
        adapter = TTADetectionAdapter(detector, entropy_margin_ratio=self.entropy_margin_ratio)
        batch = adapter.build_batch([sample.frame for sample in samples])
        trainable_model = adapter.trainable_model
        model_lock = getattr(detector, "model_lock", None)
        if model_lock is None:
            raise _TTASkip("model_lock_unavailable")
        with model_lock:
            previous_mode = bool(getattr(trainable_model, "training", False))
            module_training_state = _module_training_state(trainable_model)
            grad_state = _parameter_grad_state(trainable_model)
            try:
                das_trainer = self._ensure_das_trainer_locked(trainable_model)
                self._select_trainable_parameters(trainable_model)
                trainable_params = [
                    param for param in trainable_model.parameters() if param.requires_grad
                ]
                trainable_param_count = sum(int(param.numel()) for param in trainable_params)
                if not trainable_params:
                    raise _TTASkip("no_trainable_parameters")
                optimizer = self._make_optimizer(trainable_params)
                if hasattr(trainable_model, "train"):
                    trainable_model.train(True)
                if das_trainer is not None:
                    self._probe_das_locked(adapter, batch, trainable_model, das_trainer)
                losses: list[float] = []
                for _ in range(self.tta_steps):
                    optimizer.zero_grad(set_to_none=True)
                    outputs = adapter.forward_tta_outputs(batch)
                    loss, _ = adapter.entropy_loss(outputs)
                    if self.consistency_weight > 0.0:
                        augmented = adapter.forward_tta_outputs(batch, augment=True)
                        consistency = adapter.consistency_loss(outputs, augmented)
                        if consistency is not None:
                            loss = loss + (self.consistency_weight * consistency)
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach().item()))
                model_version_before = str(getattr(self._edge, "model_version", "0") or "0")
                model_version_after = _next_surgeon_version(model_version_before)
                self._edge.model_version = model_version_after
                return {
                    "batch_size": len(batch),
                    "loss": losses[-1] if losses else 0.0,
                    "das_enabled": das_trainer is not None,
                    "trainable_param_count": trainable_param_count,
                    "model_version_before": model_version_before,
                    "model_version_after": model_version_after,
                }
            finally:
                _clear_gradients(trainable_model)
                _restore_parameter_grad_state(trainable_model, grad_state)
                _restore_module_training_state(
                    trainable_model,
                    module_training_state,
                    previous_mode,
                )

    def _ensure_das_trainer_locked(self, model: torch.nn.Module):
        if not bool(getattr(self.das_cfg, "enabled", False)):
            return None
        if self._das_trainer is not None and self._das_model_id == id(model):
            return self._das_trainer
        self._das_trainer = apply_das_to_model(
            model,
            bn_only=bool(getattr(self.das_cfg, "bn_only", True)),
            probe_samples=int(getattr(self.das_cfg, "probe_samples", 10)),
            strategy=str(getattr(self.das_cfg, "strategy", "tgi")),
            use_spectral_entropy=bool(getattr(self.das_cfg, "use_spectral_entropy", False)),
            device=_model_device(model),
        )
        self._das_model_id = id(model)
        return self._das_trainer

    def _probe_das_locked(self, adapter, batch, model, das_trainer) -> None:
        das_trainer.deactivate_sparsity()
        _clear_gradients(model)
        outputs = adapter.forward_tta_outputs(batch)
        loss, _ = adapter.entropy_loss(outputs)
        loss.backward()
        ratios = _compute_pruning_ratios(model)
        _clear_gradients(model)
        das_trainer.activate_sparsity(ratios)

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


def _prediction_from_artifacts(artifacts: dict[str, Any]) -> dict[str, Any]:
    prediction = dict(artifacts or {})
    prediction.setdefault("boxes", prediction.get("final_detection_boxes", []))
    prediction.setdefault("labels", prediction.get("final_detection_labels", []))
    prediction.setdefault("scores", prediction.get("final_detection_scores", []))
    entropy = _finite_float(
        prediction.get("output_entropy", prediction.get("logit_entropy", prediction.get("entropy")))
    )
    if entropy is not None:
        prediction["output_entropy"] = entropy
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
    p = tensor.flatten().clamp(1.0e-8, 1.0 - 1.0e-8)
    entropy = -((p * torch.log(p)) + ((1.0 - p) * torch.log(1.0 - p)))
    return float((entropy / math.log(2.0)).mean().item())


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
        for key, mode in (
            ("pred_logits", "softmax_bg_last"),
            ("logits", "softmax_bg_last"),
            ("cls_logits", "sigmoid"),
            ("dense_logits", "softmax_bg_last"),
        ):
            value = outputs.get(key)
            if isinstance(value, torch.Tensor):
                return value, mode
    if hasattr(outputs, "logits") and isinstance(outputs.logits, torch.Tensor):
        return outputs.logits, "softmax_bg_last"
    try:
        from model_management.split_model_adapters import _extract_runtime_logits

        return _extract_runtime_logits(model, outputs)
    except Exception:
        return None, "sigmoid"


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


def _compute_pruning_ratios(model: torch.nn.Module) -> dict[str, float]:
    names: list[str] = []
    params: list[torch.Tensor] = []
    grads: list[torch.Tensor] = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        names.append(name)
        params.append(param)
        grads.append(param.grad.detach().clone())
    if not names:
        return {}
    layer_memories: dict[str, float] = {}
    memory_sum = 0.0
    for module_name, module in model.named_modules():
        if not hasattr(module, "activation_size"):
            continue
        key = f"{module_name}.weight" if module_name else "weight"
        memory = float(getattr(module, "activation_size", 1) or 1)
        layer_memories[key] = memory
        memory_sum += memory
    tgi = compute_tgi(
        params,
        grads,
        names,
        [layer_memories.get(name, 1.0) for name in names],
        memory_sum,
    )
    finite = {key: float(value) for key, value in tgi.items() if math.isfinite(float(value))}
    if not finite:
        return {}
    max_score = max(finite.values())
    if max_score <= 0.0:
        return {key: 0.0 for key in finite}
    return {key: min(1.0, max(0.0, 1.0 - value / max_score)) for key, value in finite.items()}


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
