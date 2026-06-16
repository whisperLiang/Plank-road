from __future__ import annotations

import base64
import copy
import io
import math
import random
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger

from cloud.training.baseline_workspace import resolve_training_device
from cloud.training.parameter_freeze import (
    RawFrameTrainingSample,
    apply_parameter_ratio_freeze,
    decode_training_sample,
    selected_trainable_parameters,
    unwrap_trainable_module,
)
from cloud.training.proxy_eval import _batched_predictions_from_model_output
from cloud.training.strategies.baseline_freeze import (
    _build_optimizer,
    _forward_full_model,
    _prepare_raw_batch_for_full_forward,
    run_parameter_ratio_freeze_microprofile,
)
from model_management.model_delta_payload import require_state_dict_delta_payload
from model_management.model_zoo import build_detection_model
from model_management.split_model_adapters import build_split_training_loss


@dataclass(frozen=True)
class EkyaCandidateConfig:
    config_id: str
    training_strategy: str
    trainable_param_ratio: float
    sample_fraction: float
    sample_count: int
    batch_size: int
    formal_num_epoch: int
    learning_rate: float


@dataclass(frozen=True)
class EkyaWindowSample:
    run_id: str
    baseline_method: str
    edge_id: int
    frame_id: int
    timestamp_ms: int
    model_name: str
    model_version: str
    video_source: str
    raw_frame: bytes
    edge_prediction: dict[str, Any]
    cloud_prediction: dict[str, Any]
    teacher_prediction: dict[str, Any]
    quality_metadata: dict[str, Any]
    is_keyframe: bool = True


@dataclass(frozen=True)
class EkyaReadyWindow:
    edge_id: int
    window_id: str
    run_id: str
    baseline_method: str
    model_name: str
    model_version: str
    video_source: str
    samples: tuple[EkyaWindowSample, ...]


@dataclass(frozen=True)
class MicroProfileResult:
    edge_id: int
    window_id: str
    config_id: str
    training_strategy: str
    trainable_param_ratio: float
    sample_count: int
    microprofile_epochs: int
    formal_num_epoch: int
    batch_size: int
    learning_rate: float
    proxy_metric_name: str
    proxy_metric_before: float
    proxy_metric_after_by_epoch: list[float]
    estimated_final_proxy_metric: float
    proxy_metric_gain: float
    elapsed_ms: float
    epoch_time_ms_at_full_gpu: float
    estimated_full_training_time_ms: float
    estimated_inference_penalty: float
    estimated_window_average_quality: float
    score: float
    diagnostic_loss_before: float | None = None
    diagnostic_loss_after: float | None = None


@dataclass
class CloudScheduledEkyaJob:
    edge_id: int
    window_id: str
    config_id: str
    job_id: str
    request_id: str
    base_model_version: str
    frame_ids: tuple[int, ...]
    status: str = "QUEUED"
    result_model_version: str = ""
    model_data: str = ""
    submitted_at_ms: int = 0
    finished_at_ms: int = 0


@dataclass
class EkyaCommandRecord:
    command_id: str
    edge_id: int
    job_id: str
    window_id: str
    base_model_version: str
    result_model_version: str = ""
    state: str = "pending"
    created_at_ms: int = 0
    delivered_at_ms: int = 0
    expires_at_ms: int = 0
    acked_at_ms: int = 0
    delivery_count: int = 0

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "baseline_training_job_available",
            "command_id": self.command_id,
            "job_id": self.job_id,
            "window_id": self.window_id,
            "base_model_version": self.base_model_version,
            "result_model_version": self.result_model_version,
            "expires_at_ms": int(self.expires_at_ms),
        }


class EkyaMicroProfiler:
    def __init__(
        self,
        *,
        training_config: object | Mapping[str, Any] | None,
        ekya_config: object | Mapping[str, Any] | None,
        model_weights_path: str = "",
        tinynext_input_size: int | None = None,
        model_builder: Callable[..., torch.nn.Module] | None = None,
        loss_builder: Callable[[torch.nn.Module], Callable[[Any, Any], torch.Tensor] | None]
        | None = None,
    ) -> None:
        self.training_config = _config_dict(training_config)
        self.ekya_config = _config_dict(ekya_config)
        self.model_weights_path = str(model_weights_path or "")
        self.tinynext_input_size = tinynext_input_size
        self.model_builder = model_builder or build_detection_model
        self.loss_builder = loss_builder or build_split_training_loss

    def candidate_configs(self, *, window_sample_count: int) -> list[EkyaCandidateConfig]:
        ratios = _float_list(self.ekya_config.get("trainable_param_ratios"), [0.1, 0.3, 0.5])
        fractions = _float_list(self.ekya_config.get("sample_fractions"), [0.5, 1.0])
        batch_sizes = _int_list(
            self.ekya_config.get("batch_sizes"),
            [int(self.training_config.get("batch_size", 32) or 32)],
        )
        formal_epochs = _int_list(
            self.ekya_config.get("formal_num_epochs"),
            [int(self.training_config.get("num_epoch", 50) or 50)],
        )
        learning_rates = _float_list(
            self.ekya_config.get("learning_rates"),
            [float(self.training_config.get("learning_rate", 1e-3) or 1e-3)],
        )
        candidates: list[EkyaCandidateConfig] = []
        for ratio in ratios:
            for fraction in fractions:
                sample_count = max(1, int(math.ceil(max(1, window_sample_count) * fraction)))
                sample_count = min(max(1, window_sample_count), sample_count)
                for batch_size in batch_sizes:
                    for epoch_count in formal_epochs:
                        for lr_index, learning_rate in enumerate(learning_rates):
                            config_id = (
                                f"freeze-r{ratio:.6g}-s{sample_count}-"
                                f"b{batch_size}-e{epoch_count}-lr{lr_index}"
                            )
                            candidates.append(
                                EkyaCandidateConfig(
                                    config_id=config_id,
                                    training_strategy="freeze",
                                    trainable_param_ratio=float(ratio),
                                    sample_fraction=float(fraction),
                                    sample_count=int(sample_count),
                                    batch_size=int(batch_size),
                                    formal_num_epoch=int(epoch_count),
                                    learning_rate=float(learning_rate),
                                )
                            )
        candidates.sort(
            key=lambda item: (
                int(item.sample_count),
                float(item.trainable_param_ratio),
                int(item.formal_num_epoch),
                int(item.batch_size),
                item.config_id,
            )
        )
        limit = max(1, int(self.ekya_config.get("max_microprofile_configs", 8) or 8))
        return candidates[:limit]

    def profile_window(
        self,
        window: EkyaReadyWindow,
        *,
        base_model_update_model_data: str = "",
    ) -> list[MicroProfileResult]:
        results: list[MicroProfileResult] = []
        candidates = self.candidate_configs(window_sample_count=len(window.samples))
        for candidate in candidates:
            result = self.profile_candidate(
                window,
                candidate,
                base_model_update_model_data=base_model_update_model_data,
            )
            if result is not None:
                results.append(result)
        return results

    def profile_candidate(
        self,
        window: EkyaReadyWindow,
        candidate: EkyaCandidateConfig,
        *,
        base_model_update_model_data: str = "",
    ) -> MicroProfileResult | None:
        selected_window_samples = select_window_samples(
            window.samples,
            sample_count=candidate.sample_count,
            seed=f"{window.window_id}:{candidate.config_id}:formal",
        )
        max_microprofile_samples = max(
            1,
            int(self.training_config.get("microprofile_max_samples", 16) or 16),
        )
        microprofile_samples = select_window_samples(
            selected_window_samples,
            sample_count=min(len(selected_window_samples), max_microprofile_samples),
            seed=f"{window.window_id}:{candidate.config_id}:microprofile",
        )
        teacher_objects = count_teacher_objects(microprofile_samples)
        min_teacher_objects = max(1, int(self.ekya_config.get("min_teacher_objects", 1) or 1))
        if teacher_objects < min_teacher_objects:
            logger.info(
                "ekya_schedule_skip edge={} window={} reason=proxy_metric_unavailable",
                window.edge_id,
                window.window_id,
            )
            return None

        training_samples = [
            decode_training_sample(
                frame_id=sample.frame_id,
                raw_frame=sample.raw_frame,
                target=sample.teacher_prediction,
            )
            for sample in microprofile_samples
        ]
        device = resolve_training_device(self.training_config.get("device", "auto"))
        model = self._build_model(window.model_name, device=device)
        _load_base_model_update(model, base_model_update_model_data, device=device)
        trainable_module = unwrap_trainable_module(model, model_name=window.model_name)
        trainable_module.to(device)
        freeze_summary = apply_parameter_ratio_freeze(
            trainable_module,
            candidate.trainable_param_ratio,
        )
        selected = selected_trainable_parameters(freeze_summary)
        optimizer = _build_optimizer(
            [parameter for _name, parameter in selected],
            learning_rate=candidate.learning_rate,
            optimizer_name=str(self.training_config.get("optimizer_name", "adam") or "adam"),
            weight_decay=float(self.training_config.get("weight_decay", 0.0) or 0.0),
        )
        loss_fn = self.loss_builder(model)
        proxy_before = evaluate_teacher_agreement_f1(
            model,
            trainable_module,
            training_samples,
            batch_size=candidate.batch_size,
            device=device,
            iou_threshold=float(self.ekya_config.get("teacher_agreement_iou_threshold", 0.5)),
            confidence_threshold=float(
                self.ekya_config.get("teacher_agreement_confidence_threshold", 0.0)
            ),
            min_teacher_objects=min_teacher_objects,
        )
        if proxy_before is None:
            logger.info(
                "ekya_schedule_skip edge={} window={} reason=proxy_metric_unavailable",
                window.edge_id,
                window.window_id,
            )
            return None

        epochs = max(1, int(self.training_config.get("microprofile_epochs", 1) or 1))
        logger.info(
            "ekya_microprofile_start edge={} window={} config={} samples={} epochs={}",
            window.edge_id,
            window.window_id,
            candidate.config_id,
            len(microprofile_samples),
            epochs,
        )

        def evaluate_epoch(epoch: int) -> float | None:
            value = evaluate_teacher_agreement_f1(
                model,
                trainable_module,
                training_samples,
                batch_size=candidate.batch_size,
                device=device,
                iou_threshold=float(self.ekya_config.get("teacher_agreement_iou_threshold", 0.5)),
                confidence_threshold=float(
                    self.ekya_config.get("teacher_agreement_confidence_threshold", 0.0)
                ),
                min_teacher_objects=min_teacher_objects,
            )
            if value is not None:
                logger.info(
                    "ekya_microprofile_epoch edge={} window={} config={} epoch={} "
                    "proxy_metric=teacher_agreement_f1 value={}",
                    window.edge_id,
                    window.window_id,
                    candidate.config_id,
                    epoch,
                    value,
                )
            return value

        metrics = run_parameter_ratio_freeze_microprofile(
            model=model,
            trainable_module=trainable_module,
            samples=training_samples,
            batch_size=candidate.batch_size,
            epochs=epochs,
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer,
            evaluate_epoch=evaluate_epoch,
        )
        elapsed_ms = float(metrics.get("microprofile_time_sec", 0.0) or 0.0) * 1000.0
        after_values = [
            float(value)
            for value in list(metrics.get("proxy_metric_after_by_epoch", []) or [])
            if value is not None
        ]
        if not after_values:
            logger.info(
                "ekya_schedule_skip edge={} window={} reason=proxy_metric_unavailable",
                window.edge_id,
                window.window_id,
            )
            return None
        final_observed = float(after_values[-1])
        estimated_final = _estimate_final_metric(
            proxy_before,
            final_observed,
            microprofile_epochs=epochs,
            formal_epochs=candidate.formal_num_epoch,
        )
        gain = estimated_final - proxy_before
        epoch_time_ms = elapsed_ms / max(1, epochs)
        estimated_full_training_time_ms = estimate_full_training_time_ms(
            epoch_time_ms=epoch_time_ms,
            formal_epochs=candidate.formal_num_epoch,
            formal_sample_count=candidate.sample_count,
            microprofile_sample_count=len(microprofile_samples),
        )
        inference_penalty = estimate_inference_penalty(
            estimated_full_training_time_ms=estimated_full_training_time_ms,
            window_samples=len(window.samples),
        )
        estimated_window_average_quality = max(0.0, min(1.0, estimated_final - inference_penalty))
        score = estimated_window_average_quality - proxy_before
        result = MicroProfileResult(
            edge_id=window.edge_id,
            window_id=window.window_id,
            config_id=candidate.config_id,
            training_strategy=candidate.training_strategy,
            trainable_param_ratio=candidate.trainable_param_ratio,
            sample_count=candidate.sample_count,
            microprofile_epochs=epochs,
            formal_num_epoch=candidate.formal_num_epoch,
            batch_size=candidate.batch_size,
            learning_rate=candidate.learning_rate,
            proxy_metric_name="teacher_agreement_f1",
            proxy_metric_before=proxy_before,
            proxy_metric_after_by_epoch=after_values,
            estimated_final_proxy_metric=estimated_final,
            proxy_metric_gain=gain,
            elapsed_ms=elapsed_ms,
            epoch_time_ms_at_full_gpu=epoch_time_ms,
            estimated_full_training_time_ms=estimated_full_training_time_ms,
            estimated_inference_penalty=inference_penalty,
            estimated_window_average_quality=estimated_window_average_quality,
            score=score,
            diagnostic_loss_before=_optional_float(metrics.get("loss_before")),
            diagnostic_loss_after=_optional_float(metrics.get("final_loss")),
        )
        logger.info(
            "ekya_microprofile_done edge={} window={} config={} "
            "estimated_final_proxy_metric={} estimated_full_training_time_ms={} "
            "estimated_window_average_quality={} score={}",
            window.edge_id,
            window.window_id,
            candidate.config_id,
            result.estimated_final_proxy_metric,
            result.estimated_full_training_time_ms,
            result.estimated_window_average_quality,
            result.score,
        )
        return result

    def _build_model(self, model_name: str, *, device: torch.device) -> torch.nn.Module:
        kwargs: dict[str, Any] = {}
        if self.tinynext_input_size is not None and str(model_name).lower().startswith("tinynext"):
            kwargs["tinynext_input_size"] = int(self.tinynext_input_size)
        model = self.model_builder(
            str(model_name),
            pretrained=True,
            device=device,
            weights_path=self.model_weights_path or None,
            **kwargs,
        )
        if not isinstance(model, torch.nn.Module):
            raise RuntimeError(f"model_builder returned non-module: {type(model)!r}")
        model.to(device)
        return model


class EkyaCentralScheduler:
    def __init__(
        self,
        *,
        ready_windows: Callable[[], Iterable[EkyaReadyWindow]],
        profile_window: Callable[[EkyaReadyWindow], list[MicroProfileResult]],
        submit_training: Callable[[EkyaReadyWindow, MicroProfileResult], str | None],
        mark_skip: Callable[[EkyaReadyWindow, str], None] | None = None,
        active_training_count: Callable[[], int] | None = None,
        service_state: Callable[[], Mapping[str, float]] | None = None,
        ekya_config: object | Mapping[str, Any] | None = None,
    ) -> None:
        self.ready_windows = ready_windows
        self.profile_window = profile_window
        self.submit_training = submit_training
        self.mark_skip = mark_skip or (lambda _window, _reason: None)
        self.active_training_count = active_training_count or (lambda: 0)
        self.service_state = service_state or (lambda: {})
        self.ekya_config = _config_dict(ekya_config)

    def run_once(self) -> MicroProfileResult | None:
        if int(self.active_training_count()) > 0:
            return None
        candidates: list[tuple[EkyaReadyWindow, MicroProfileResult]] = []
        for window in list(self.ready_windows()):
            results = self.profile_window(window)
            if not results:
                self.mark_skip(window, "proxy_metric_unavailable")
                continue
            viable = [result for result in results if self._is_viable(result)]
            if not viable:
                self.mark_skip(window, "no_candidate_improves_window_quality")
                continue
            best = max(
                viable,
                key=lambda result: (
                    result.score,
                    result.estimated_window_average_quality,
                ),
            )
            candidates.append((window, best))
        if not candidates:
            return None
        selected_window, selected_result = max(
            candidates,
            key=lambda item: (
                item[1].score,
                item[1].estimated_window_average_quality,
                -item[0].edge_id,
            ),
        )
        job_id = self.submit_training(selected_window, selected_result)
        if not job_id:
            self.mark_skip(selected_window, "training_job_rejected")
            return None
        logger.info(
            "ekya_schedule_select edge={} window={} config={} score={} reason={}",
            selected_window.edge_id,
            selected_window.window_id,
            selected_result.config_id,
            selected_result.score,
            "max_window_average_quality",
        )
        return selected_result

    def _is_viable(self, result: MicroProfileResult) -> bool:
        if result.score <= 0.0:
            return False
        min_quality = float(self.ekya_config.get("min_inference_quality", 0.0) or 0.0)
        if min_quality > 0.0 and result.estimated_window_average_quality < min_quality:
            return False
        state = self.service_state()
        max_latency = float(self.ekya_config.get("max_cloud_inference_latency_ms", 0.0) or 0.0)
        observed_latency = float(state.get("cloud_inference_latency_ms", 0.0) or 0.0)
        if max_latency > 0.0 and observed_latency > max_latency:
            return False
        min_fps = float(self.ekya_config.get("min_cloud_inference_fps", 0.0) or 0.0)
        if min_fps > 0.0 and float(state.get("cloud_inference_fps", 0.0) or 0.0) < min_fps:
            return False
        return True


def select_window_samples(
    samples: Iterable[EkyaWindowSample],
    *,
    sample_count: int,
    seed: object,
) -> list[EkyaWindowSample]:
    sample_list = list(samples)
    count = min(len(sample_list), max(1, int(sample_count)))
    if count >= len(sample_list):
        return list(sample_list)
    rng = random.Random(str(seed))
    indices = sorted(rng.sample(range(len(sample_list)), count))
    return [sample_list[index] for index in indices]


def count_teacher_objects(samples: Iterable[EkyaWindowSample]) -> int:
    total = 0
    for sample in samples:
        boxes = sample.teacher_prediction.get("boxes") or []
        try:
            total += len(boxes)
        except TypeError:
            continue
    return total


def evaluate_teacher_agreement_f1(
    model: torch.nn.Module,
    trainable_module: torch.nn.Module,
    samples: list[RawFrameTrainingSample],
    *,
    batch_size: int,
    device: torch.device,
    iou_threshold: float,
    confidence_threshold: float,
    min_teacher_objects: int,
) -> float | None:
    teacher_objects = sum(len(list(sample.target.get("boxes") or [])) for sample in samples)
    if teacher_objects < max(1, int(min_teacher_objects)):
        return None
    predictions = _predict_samples(
        model,
        trainable_module,
        samples,
        batch_size=batch_size,
        device=device,
    )
    tp = fp = fn = 0
    for sample, prediction in zip(samples, predictions):
        sample_tp, sample_fp, sample_fn = teacher_agreement_counts(
            prediction,
            sample.target,
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold,
        )
        tp += sample_tp
        fp += sample_fp
        fn += sample_fn
    denominator = (2 * tp) + fp + fn
    if denominator <= 0:
        return None
    return float((2 * tp) / denominator)


def teacher_agreement_counts(
    prediction: Mapping[str, Any],
    teacher: Mapping[str, Any],
    *,
    iou_threshold: float,
    confidence_threshold: float,
) -> tuple[int, int, int]:
    teacher_boxes = _boxes(teacher.get("boxes"))
    teacher_labels = _labels(teacher.get("labels"), len(teacher_boxes))
    pred_boxes = _boxes(prediction.get("boxes"))
    pred_labels = _labels(prediction.get("labels"), len(pred_boxes))
    pred_scores = _scores(prediction.get("scores"), len(pred_boxes))
    kept_indices = [
        index for index, score in enumerate(pred_scores) if float(score) >= confidence_threshold
    ]
    matched_teacher: set[int] = set()
    tp = 0
    for pred_index in kept_indices:
        best_teacher = -1
        best_iou = 0.0
        for teacher_index, teacher_box in enumerate(teacher_boxes):
            if teacher_index in matched_teacher:
                continue
            if teacher_index >= len(teacher_labels) or pred_index >= len(pred_labels):
                continue
            if int(teacher_labels[teacher_index]) != int(pred_labels[pred_index]):
                continue
            iou = box_iou(pred_boxes[pred_index], teacher_box)
            if iou > best_iou:
                best_iou = iou
                best_teacher = teacher_index
        if best_teacher >= 0 and best_iou >= float(iou_threshold):
            matched_teacher.add(best_teacher)
            tp += 1
    fp = max(0, len(kept_indices) - tp)
    fn = max(0, len(teacher_boxes) - tp)
    return tp, fp, fn


def box_iou(first: Iterable[float], second: Iterable[float]) -> float:
    a = [float(value) for value in list(first)[:4]]
    b = [float(value) for value in list(second)[:4]]
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def estimate_full_training_time_ms(
    *,
    epoch_time_ms: float,
    formal_epochs: int,
    formal_sample_count: int,
    microprofile_sample_count: int,
) -> float:
    sample_scale = max(1.0, float(formal_sample_count) / max(1.0, float(microprofile_sample_count)))
    return max(0.0, float(epoch_time_ms)) * max(1, int(formal_epochs)) * sample_scale


def estimate_inference_penalty(
    *,
    estimated_full_training_time_ms: float,
    window_samples: int,
) -> float:
    denominator = max(1.0, float(window_samples) * 1000.0)
    return min(0.5, max(0.0, float(estimated_full_training_time_ms) / denominator) * 0.05)


def _predict_samples(
    model: torch.nn.Module,
    trainable_module: torch.nn.Module,
    samples: list[RawFrameTrainingSample],
    *,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, list]]:
    predictions: list[dict[str, list]] = []
    model.eval()
    trainable_module.eval()
    for batch in _batches(samples, max(1, int(batch_size))):
        prepared = _prepare_raw_batch_for_full_forward(
            model,
            trainable_module,
            batch,
            device=device,
        )
        outputs = _forward_full_model(model, trainable_module, prepared)
        predictions.extend(
            _batched_predictions_from_model_output(
                outputs,
                batch_size=len(batch),
                threshold_low=0.0,
                threshold_high=0.0,
            )
        )
    return predictions


def _batches(samples: list[RawFrameTrainingSample], batch_size: int):
    for index in range(0, len(samples), max(1, int(batch_size))):
        yield samples[index : index + max(1, int(batch_size))]


def _estimate_final_metric(
    before: float,
    observed_final: float,
    *,
    microprofile_epochs: int,
    formal_epochs: int,
) -> float:
    gain_per_epoch = (float(observed_final) - float(before)) / max(1, int(microprofile_epochs))
    estimated = float(before) + gain_per_epoch * max(1, int(formal_epochs))
    return max(0.0, min(1.0, estimated))


def _load_base_model_update(
    model: torch.nn.Module,
    model_data: str,
    *,
    device: torch.device,
) -> None:
    if not model_data:
        return
    payload = require_state_dict_delta_payload(
        torch.load(
            io.BytesIO(base64.b64decode(str(model_data))),
            map_location=device,
            weights_only=False,
        )
    )
    model.load_state_dict(dict(payload["state_dict"]), strict=False)


def _boxes(value: object) -> list[list[float]]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and not isinstance(value[0], (list, tuple))
    ):
        value = [value]
    result = []
    for item in list(value or []):
        if hasattr(item, "detach"):
            item = item.detach().cpu().tolist()
        elif hasattr(item, "tolist"):
            item = item.tolist()
        values = list(item or [])
        if len(values) >= 4:
            result.append([float(v) for v in values[:4]])
    return result


def _labels(value: object, expected: int) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        value = [value]
    result = [int(item) for item in list(value or [])]
    return result[:expected]


def _scores(value: object, expected: int) -> list[float]:
    if value is None:
        return [1.0 for _ in range(expected)]
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        value = [value]
    result = [float(item) for item in list(value or [])]
    if len(result) < expected:
        result.extend([1.0] * (expected - len(result)))
    return result[:expected]


def _config_dict(config: object | Mapping[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    result: dict[str, Any] = {}
    for name in dir(config):
        if name.startswith("_"):
            continue
        try:
            value = getattr(config, name)
        except Exception:
            continue
        if callable(value):
            continue
        result[name] = copy.deepcopy(value)
    return result


def _float_list(value: object, default: list[float]) -> list[float]:
    values = list(value or [])
    if not values:
        values = list(default)
    return [float(item) for item in values]


def _int_list(value: object, default: list[int]) -> list[int]:
    values = list(value or [])
    if not values:
        values = list(default)
    return [int(item) for item in values]


def _optional_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
