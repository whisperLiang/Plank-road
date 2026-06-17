from __future__ import annotations

import base64
import copy
import gc
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
    build_baseline_freeze_loss,
    run_parameter_ratio_freeze_microprofile,
)
from model_management.model_delta_payload import require_state_dict_delta_payload
from model_management.model_zoo import build_detection_model
from model_management.split_model_adapters import postprocess_split_runtime_output

EKYA_STATUS_COLLECTING = "COLLECTING"
EKYA_STATUS_LABEL_PENDING = "LABEL_PENDING"
EKYA_STATUS_LABELING = "LABELING"
EKYA_STATUS_LABELED = "LABELED"
EKYA_STATUS_MICROPROFILING = "MICROPROFILING"
EKYA_STATUS_TRAINING = "TRAINING"
EKYA_STATUS_SUCCEEDED = "SUCCEEDED"
EKYA_STATUS_FAILED = "FAILED"
EKYA_STATUS_SKIPPED = "SKIPPED"

EKYA_TERMINAL_WINDOW_STATUSES = {
    EKYA_STATUS_SUCCEEDED,
    EKYA_STATUS_FAILED,
    EKYA_STATUS_SKIPPED,
}

EKYA_FAILURE_STAGE_TEACHER_ANNOTATION = "teacher_annotation"
EKYA_FAILURE_STAGE_MICROPROFILE = "microprofile"
EKYA_FAILURE_STAGE_TRAINING_SUBMISSION = "training_submission"


class EkyaHeavyLaneBusy(RuntimeError):
    """Raised when shared heavy-GPU work should be retried by the scheduler."""

    retryable = True


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
class EkyaWindowState:
    status: str
    failure_stage: str = ""
    failure_reason: str = ""
    updated_at_ms: int = 0


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
    result_id: str = ""
    base_model_version: str = "0"


@dataclass
class CloudScheduledEkyaJob:
    edge_id: int
    window_id: str
    config_id: str
    job_id: str
    request_id: str
    base_model_version: str
    frame_ids: tuple[int, ...]
    microprofile_result_id: str = ""
    status: str = "QUEUED"
    result_model_version: str = ""
    model_data: str = ""
    submitted_at_ms: int = 0
    finished_at_ms: int = 0


@dataclass
class EkyaCommandRecord:
    command_id: str
    run_id: str
    baseline_method: str
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
            "run_id": self.run_id,
            "baseline_method": self.baseline_method,
            "edge_id": int(self.edge_id),
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
        self.loss_builder = loss_builder or build_baseline_freeze_loss
        self._window_skip_reasons: dict[tuple[int, str], list[str]] = {}

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
        self._window_skip_reasons[(int(window.edge_id), str(window.window_id))] = []
        results: list[MicroProfileResult] = []
        candidates = self.candidate_configs(window_sample_count=len(window.samples))
        for candidate in candidates:
            try:
                result = self.profile_candidate(
                    window,
                    candidate,
                    base_model_update_model_data=base_model_update_model_data,
                )
            except Exception as exc:
                reason = "microprofile_failed"
                if _is_cuda_oom(exc):
                    reason = "microprofile_oom"
                logger.warning(
                    "ekya_schedule_skip edge={} window={} config={} "
                    "reason={} error={}",
                    window.edge_id,
                    window.window_id,
                    candidate.config_id,
                    reason,
                    exc,
                )
                self._record_skip_reason(window, "microprofile_failed")
                _release_cuda_cache()
                continue
            if result is not None:
                results.append(result)
        return results

    def skip_reason(self, window: EkyaReadyWindow) -> str:
        reasons = self._window_skip_reasons.get((int(window.edge_id), str(window.window_id)), [])
        priority = {
            "insufficient_samples": 0,
            "teacher_labels_unavailable": 1,
            "proxy_metric_unavailable": 2,
            "microprofile_failed": 3,
        }
        valid = [reason for reason in reasons if reason in priority]
        if not valid:
            return ""
        return min(valid, key=lambda reason: priority[reason])

    def profile_candidate(
        self,
        window: EkyaReadyWindow,
        candidate: EkyaCandidateConfig,
        *,
        base_model_update_model_data: str = "",
    ) -> MicroProfileResult | None:
        if not window.samples:
            logger.info(
                "ekya_schedule_skip edge={} window={} reason=insufficient_samples",
                window.edge_id,
                window.window_id,
            )
            self._record_skip_reason(window, "insufficient_samples")
            return None
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
        if not microprofile_samples or not any(sample.raw_frame for sample in microprofile_samples):
            logger.info(
                "ekya_schedule_skip edge={} window={} reason=insufficient_samples",
                window.edge_id,
                window.window_id,
            )
            self._record_skip_reason(window, "insufficient_samples")
            return None
        if not any(sample.teacher_prediction for sample in microprofile_samples):
            logger.info(
                "ekya_schedule_skip edge={} window={} reason=teacher_labels_unavailable",
                window.edge_id,
                window.window_id,
            )
            self._record_skip_reason(window, "teacher_labels_unavailable")
            return None
        teacher_objects = count_teacher_objects(microprofile_samples)
        min_teacher_objects = max(1, int(self.ekya_config.get("min_teacher_objects", 1) or 1))
        if teacher_objects < min_teacher_objects:
            logger.info(
                "ekya_schedule_skip edge={} window={} reason=proxy_metric_unavailable",
                window.edge_id,
                window.window_id,
            )
            self._record_skip_reason(window, "proxy_metric_unavailable")
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
        model = trainable_module = optimizer = loss_fn = None
        metrics: dict[str, Any] = {}
        try:
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
                self._record_skip_reason(window, "proxy_metric_unavailable")
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
                    iou_threshold=float(
                        self.ekya_config.get("teacher_agreement_iou_threshold", 0.5)
                    ),
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
                self._record_skip_reason(window, "proxy_metric_unavailable")
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
            estimated_window_average_quality = max(
                0.0,
                min(1.0, estimated_final - inference_penalty),
            )
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
                result_id=microprofile_result_id(window, candidate),
                base_model_version=str(window.model_version or "0"),
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
        finally:
            del metrics
            _cleanup_microprofile_objects(model, trainable_module, optimizer, loss_fn)

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

    def _record_skip_reason(self, window: EkyaReadyWindow, reason: str) -> None:
        key = (int(window.edge_id), str(window.window_id))
        self._window_skip_reasons.setdefault(key, []).append(str(reason))


class EkyaCentralScheduler:
    def __init__(
        self,
        *,
        ready_windows: Callable[[], Iterable[EkyaReadyWindow]],
        profile_window: Callable[[EkyaReadyWindow], list[MicroProfileResult]],
        submit_training: Callable[[EkyaReadyWindow, MicroProfileResult], str | None],
        label_pending_windows: Callable[[], Iterable[EkyaReadyWindow]] | None = None,
        annotate_window: Callable[[EkyaReadyWindow], bool] | None = None,
        mark_skip: Callable[[EkyaReadyWindow, str], None] | None = None,
        profile_skip_reason: Callable[[EkyaReadyWindow], str] | None = None,
        active_training_count: Callable[[], int] | None = None,
        service_state: Callable[[], Mapping[str, float]] | None = None,
        ekya_config: object | Mapping[str, Any] | None = None,
    ) -> None:
        self.ready_windows = ready_windows
        self.profile_window = profile_window
        self.submit_training = submit_training
        self.label_pending_windows = label_pending_windows or (lambda: [])
        self.annotate_window = annotate_window
        self.mark_skip = mark_skip or (lambda _window, _reason: None)
        self.profile_skip_reason = profile_skip_reason or (lambda _window: "")
        self.active_training_count = active_training_count or (lambda: 0)
        self.service_state = service_state or (lambda: {})
        self.ekya_config = _config_dict(ekya_config)

    def run_once(self) -> MicroProfileResult | None:
        if int(self.active_training_count()) > 0:
            return None

        pending_windows = list(self.label_pending_windows())
        if pending_windows and self.annotate_window is not None:
            self.annotate_window(pending_windows[0])
            return None

        windows = list(self.ready_windows())
        if not windows:
            return None
        selected_window = windows[0]
        try:
            results = self.profile_window(selected_window)
        except EkyaHeavyLaneBusy as exc:
            logger.info(
                "ekya_schedule_defer edge={} window={} reason=heavy_gpu_lease_busy error={}",
                selected_window.edge_id,
                selected_window.window_id,
                exc,
            )
            return None
        except Exception as exc:
            logger.warning(
                "ekya_schedule_skip edge={} window={} reason=microprofile_failed error={}",
                selected_window.edge_id,
                selected_window.window_id,
                exc,
            )
            self.mark_skip(selected_window, "microprofile_failed")
            return None
        if not results:
            reason = self.profile_skip_reason(selected_window) or "proxy_metric_unavailable"
            self.mark_skip(selected_window, reason)
            return None
        evaluated = [
            (result, self._viability_reason(selected_window, result))
            for result in results
        ]
        viable = [result for result, reason in evaluated if reason == ""]
        if not viable:
            reasons = {reason for _result, reason in evaluated if reason}
            if "service_quality_constraint_failed" in reasons:
                self.mark_skip(selected_window, "service_quality_constraint_failed")
            elif "microprofile_failed" in reasons:
                self.mark_skip(selected_window, "microprofile_failed")
            else:
                self.mark_skip(selected_window, "no_candidate_improves_window_quality")
            return None
        selected_result = max(
            viable,
            key=lambda result: (
                result.score,
                result.estimated_window_average_quality,
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

    def _viability_reason(self, window: EkyaReadyWindow, result: MicroProfileResult) -> str:
        if int(result.edge_id) != int(window.edge_id):
            return "microprofile_failed"
        if str(result.window_id) != str(window.window_id):
            return "microprofile_failed"
        if str(result.training_strategy) != "freeze":
            return "microprofile_failed"
        if str(result.proxy_metric_name) != "teacher_agreement_f1":
            return "proxy_metric_unavailable"
        if str(result.base_model_version or "0") != str(window.model_version or "0"):
            return "microprofile_failed"
        allow_zero_gain = bool(self.ekya_config.get("allow_zero_gain_training", True))
        if float(result.score) < 0.0 or (
            float(result.score) <= 0.0 and not allow_zero_gain
        ):
            return "no_candidate_improves_window_quality"
        min_quality = float(self.ekya_config.get("min_inference_quality", 0.0) or 0.0)
        if min_quality > 0.0 and result.estimated_window_average_quality < min_quality:
            return "service_quality_constraint_failed"
        state = self.service_state()
        max_latency = float(self.ekya_config.get("max_cloud_inference_latency_ms", 0.0) or 0.0)
        observed_latency = float(state.get("cloud_inference_latency_ms", 0.0) or 0.0)
        if max_latency > 0.0 and observed_latency > max_latency:
            return "service_quality_constraint_failed"
        min_fps = float(self.ekya_config.get("min_cloud_inference_fps", 0.0) or 0.0)
        if min_fps > 0.0 and float(state.get("cloud_inference_fps", 0.0) or 0.0) < min_fps:
            return "service_quality_constraint_failed"
        return ""


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


def microprofile_result_id(window: EkyaReadyWindow, candidate: EkyaCandidateConfig) -> str:
    return (
        f"ekya-microprofile:{window.run_id}:{window.edge_id}:"
        f"{window.window_id}:{candidate.config_id}:{window.model_version or '0'}"
    )


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
    metric = float((2 * tp) / denominator)
    if metric == 0.0:
        predicted_objects = sum(
            len(list(prediction.get("boxes") or [])) for prediction in predictions
        )
        logger.info(
            "ekya_proxy_eval_zero samples={} teacher_objects={} predicted_objects={} "
            "tp={} fp={} fn={}",
            len(samples),
            teacher_objects,
            predicted_objects,
            tp,
            fp,
            fn,
        )
    return metric


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
    with torch.inference_mode():
        for batch in _batches(samples, max(1, int(batch_size))):
            prepared = _prepare_raw_batch_for_full_forward(
                model,
                trainable_module,
                batch,
                device=device,
            )
            outputs = _forward_full_model(model, trainable_module, prepared)
            batch_predictions = _batched_predictions_from_model_output(
                outputs,
                batch_size=len(batch),
                threshold_low=0.0,
                threshold_high=0.0,
            )
            if _looks_like_raw_detection_output(outputs):
                postprocessed = _postprocess_raw_detection_output(
                    model,
                    outputs,
                    prepared=prepared,
                    batch=batch,
                )
                if postprocessed is not None:
                    batch_predictions = postprocessed
            predictions.extend(batch_predictions)
    return predictions


def _looks_like_raw_detection_output(outputs: Any) -> bool:
    if isinstance(outputs, Mapping):
        if "pred_boxes" in outputs and (
            "pred_logits" in outputs or "logits" in outputs
        ):
            return True
    if hasattr(outputs, "pred_boxes") and (
        hasattr(outputs, "pred_logits") or hasattr(outputs, "logits")
    ):
        return True
    # RF-DETR and several DETR-style internals return tuple outputs before
    # wrapper postprocessing converts them into boxes/labels/scores.
    return isinstance(outputs, tuple)


def _postprocess_raw_detection_output(
    model: torch.nn.Module,
    outputs: Any,
    *,
    prepared: Any,
    batch: list[RawFrameTrainingSample],
) -> list[dict[str, list]] | None:
    if not batch:
        return []
    predictions: list[dict[str, list]] = []
    for index, sample in enumerate(batch):
        try:
            decoded = postprocess_split_runtime_output(
                model,
                _slice_batch_item(outputs, index),
                threshold=0.0,
                model_input=_slice_batch_item(prepared.model_inputs, index),
                orig_image=sample.image_bgr,
            )
        except Exception as exc:
            logger.debug(
                "ekya_microprofile_postprocess_failed model_type={} index={} error={}",
                type(model).__name__,
                index,
                exc,
            )
            return None
        sample_predictions = _batched_predictions_from_model_output(
            decoded,
            batch_size=1,
            threshold_low=0.0,
            threshold_high=0.0,
        )
        if len(sample_predictions) != 1:
            logger.debug(
                "ekya_microprofile_postprocess_mismatch index={} predictions={}",
                index,
                len(sample_predictions),
            )
            return None
        predictions.extend(sample_predictions)
    return predictions


def _slice_batch_item(value: Any, index: int) -> Any:
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value
        return value[index : index + 1]
    if isinstance(value, Mapping):
        return {key: _slice_batch_item(item, index) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_slice_batch_item(item, index) for item in value)
    if isinstance(value, list):
        return [_slice_batch_item(item, index) for item in value]
    return value


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


def _cleanup_microprofile_objects(*objects: object) -> None:
    for item in objects:
        try:
            if hasattr(item, "zero_grad"):
                item.zero_grad(set_to_none=True)
        except Exception:
            pass
    gc.collect()
    _release_cuda_cache()


def _release_cuda_cache() -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text and ("cuda" in text or "gpu" in text)


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
