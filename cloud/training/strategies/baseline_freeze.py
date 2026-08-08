from __future__ import annotations

import base64
import copy
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import cv2
import torch
from loguru import logger
from torchvision.transforms import functional as F

from cloud.model_update import serialize_model_update
from cloud.training.baseline_workspace import (
    load_baseline_manifest,
    model_builder_kwargs,
    resolve_training_device,
    samples_from_baseline_manifest,
)
from cloud.training.parameter_freeze import (
    RawFrameTrainingSample,
    apply_parameter_ratio_freeze,
    selected_trainable_parameters,
    unwrap_trainable_module,
)
from model_management.detection_box_projection import ORIGINAL_XYXY
from model_management.model_delta_payload import require_state_dict_delta_payload
from model_management.model_zoo import build_detection_model
from model_management.split_model_adapters import (
    build_split_training_loss,
    get_split_runtime_input_resize_mode,
    prepare_split_runtime_input,
)


@dataclass(frozen=True)
class _PreparedBatch:
    model_inputs: Any
    raw_image_tensors: list[torch.Tensor]
    targets: list[dict[str, Any]]


class CloudBaselineFreezeTrainingStrategy:
    name = "freeze"

    def __init__(
        self,
        *,
        learner=None,
        model_builder: Callable[..., torch.nn.Module] | None = None,
        update_serializer: Callable[..., bytes] | None = None,
        loss_builder: Callable[[torch.nn.Module], Callable[[Any, Any], torch.Tensor] | None]
        | None = None,
    ) -> None:
        self.learner = learner
        self.model_builder = model_builder or build_detection_model
        self.update_serializer = update_serializer or serialize_model_update
        self.loss_builder = loss_builder or build_split_training_loss

    def train_from_workspace(
        self,
        workspace: str | Path,
        *,
        base_model_version: str = "0",
        result_model_version: str = "1",
    ) -> dict[str, Any]:
        workspace_path = Path(workspace)
        manifest = load_baseline_manifest(workspace_path)
        if manifest.get("training_strategy") != self.name:
            raise RuntimeError(f"freeze strategy received {manifest.get('training_strategy')!r}")

        training_cfg = dict(manifest.get("training_config") or {})
        ratio = _trainable_param_ratio(training_cfg)
        logger.info(
            "[BaselineTraining] strategy=freeze trainable_param_ratio={}",
            ratio,
        )
        device = resolve_training_device(training_cfg.get("device", "auto"))
        model_name = str(manifest.get("model_name", "") or "")
        if not model_name:
            raise RuntimeError("baseline trigger manifest is missing model_name")

        model = self.model_builder(
            model_name,
            pretrained=True,
            device=device,
            weights_path=str(manifest.get("weights_path", "") or "") or None,
            **model_builder_kwargs(manifest),
        )
        if not isinstance(model, torch.nn.Module):
            raise RuntimeError(f"model_builder returned non-module: {type(model)!r}")
        model.to(device)
        _load_optional_base_model_update(
            model,
            workspace_path=workspace_path,
            manifest=manifest,
            device=device,
        )

        trainable_module = unwrap_trainable_module(model, model_name=model_name)
        trainable_module.to(device)
        freeze_summary = apply_parameter_ratio_freeze(trainable_module, ratio)
        selected = selected_trainable_parameters(freeze_summary)

        learning_rate = float(training_cfg.get("learning_rate", 1e-3) or 1e-3)
        optimizer_name = str(training_cfg.get("optimizer_name", "adam") or "adam")
        weight_decay = float(training_cfg.get("weight_decay", 0.0) or 0.0)
        optimizer = _build_optimizer(
            [parameter for _name, parameter in selected],
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
        )

        samples = samples_from_baseline_manifest(
            workspace_path,
            manifest,
            teacher=getattr(self.learner, "large_od", None),
            allow_edge_targets=bool(training_cfg.get("allow_edge_targets", False)),
        )
        logger.info(
            "[BaselineTraining] cloud teacher labels generated: samples={}",
            len(samples),
        )
        loss_fn = self.loss_builder(model)
        batch_size = int(training_cfg.get("batch_size", 32) or 32)
        epochs = int(training_cfg.get("num_epoch", 50) or 50)
        logger.info(
            "[BaselineTraining] training loop: samples={} epochs={} batch_size={}",
            len(samples),
            epochs,
            batch_size,
        )
        started = time.perf_counter()
        metrics = run_parameter_ratio_freeze_training(
            model=model,
            trainable_module=trainable_module,
            samples=samples,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        metrics["parameter_freeze"] = _serializable_freeze_summary(freeze_summary)
        training_elapsed_s = float(
            metrics.get("full_train_time_sec") or time.perf_counter() - started
        )
        serialization_started = time.perf_counter()

        update_bytes = self.update_serializer(
            model,
            model_name=model_name,
            checkpoint_path=str(workspace_path / "model_update" / "baseline_freeze_state.pt"),
            weights_metadata={
                "training_strategy": self.name,
                "trainable_param_ratio": ratio,
                "source_base_model_version": str(base_model_version or "0"),
                "checkpoint_model_version": str(result_model_version or "1"),
                "baseline_method": str(manifest.get("baseline_method", "")),
                "window_id": str(manifest.get("window_id", "")),
                "total_params": int(freeze_summary["total_params"]),
                "frozen_params": int(freeze_summary["frozen_params"]),
                "trainable_params": int(freeze_summary["trainable_params"]),
                "first_trainable_param": str(freeze_summary["first_trainable_param"]),
                "last_trainable_param": str(freeze_summary["last_trainable_param"]),
            },
            metadata_path=str(workspace_path / "model_update" / "baseline_freeze_metadata.json"),
        )
        encoded_model_data = base64.b64encode(update_bytes).decode("ascii")
        serialization_elapsed_s = time.perf_counter() - serialization_started
        total_elapsed_s = time.perf_counter() - started
        return {
            "success": True,
            "model_data": encoded_model_data,
            "message": (
                "[BaselineTraining] strategy=freeze "
                f"samples={len(samples)} "
                f"training_ms={training_elapsed_s * 1000.0:.3f} "
                f"serialization_ms={serialization_elapsed_s * 1000.0:.3f} "
                f"elapsed={total_elapsed_s:.3f}s"
            ),
            "metrics": metrics,
            "result_model_version": str(result_model_version or "1"),
        }


def build_baseline_freeze_loss(model: torch.nn.Module):
    return build_split_training_loss(model)


def run_parameter_ratio_freeze_training(
    *,
    model: torch.nn.Module,
    trainable_module: torch.nn.Module,
    samples: Iterable[RawFrameTrainingSample],
    batch_size: int,
    epochs: int,
    device: torch.device,
    loss_fn: Callable[[Any, Any], torch.Tensor] | None,
    optimizer: torch.optim.Optimizer,
    log_epochs: bool = True,
    epoch_log_prefix: str = "[BaselineTraining] freeze",
) -> dict[str, Any]:
    sample_list = list(samples)
    losses: list[float] = []
    epoch_losses: list[float] = []
    started = time.perf_counter()
    model.train()
    trainable_module.train()
    for epoch in range(1, int(epochs) + 1):
        epoch_losses_for_batches: list[float] = []
        for batch in _batches(sample_list, max(1, int(batch_size))):
            prepared = _prepare_raw_batch_for_full_forward(
                model,
                trainable_module,
                batch,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            outputs = _forward_full_model(
                model,
                trainable_module,
                prepared,
            )
            loss = _compute_loss(outputs, copy.deepcopy(prepared.targets), loss_fn)
            if not torch.is_tensor(loss):
                raise RuntimeError(f"baseline freeze loss returned {type(loss)!r}")
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu().item())
            losses.append(loss_value)
            epoch_losses_for_batches.append(loss_value)
        if epoch_losses_for_batches:
            avg_loss = sum(epoch_losses_for_batches) / float(len(epoch_losses_for_batches))
            epoch_losses.append(avg_loss)
            if log_epochs:
                logger.info(
                    "{} epoch {}/{} avg_loss={:.6f}.",
                    epoch_log_prefix,
                    epoch,
                    int(epochs),
                    avg_loss,
                )
        else:
            if log_epochs:
                logger.info(
                    "{} epoch {}/{} skipped: no training batches.",
                    epoch_log_prefix,
                    epoch,
                    int(epochs),
                )
    return {
        "full_train_time_sec": time.perf_counter() - started,
        "final_loss": losses[-1] if losses else None,
        "batch_count": len(losses),
        "epoch_losses": epoch_losses,
    }


def run_parameter_ratio_freeze_microprofile(
    *,
    model: torch.nn.Module,
    trainable_module: torch.nn.Module,
    samples: Iterable[RawFrameTrainingSample],
    batch_size: int,
    epochs: int,
    device: torch.device,
    loss_fn: Callable[[Any, Any], torch.Tensor] | None,
    optimizer: torch.optim.Optimizer,
    evaluate_epoch: Callable[[int], float | None],
) -> dict[str, Any]:
    sample_list = list(samples)
    losses: list[float] = []
    proxy_metric_after_by_epoch: list[float | None] = []
    started = time.perf_counter()
    model.train()
    trainable_module.train()
    for epoch in range(1, int(epochs) + 1):
        for batch in _batches(sample_list, max(1, int(batch_size))):
            prepared = _prepare_raw_batch_for_full_forward(
                model,
                trainable_module,
                batch,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            outputs = _forward_full_model(
                model,
                trainable_module,
                prepared,
            )
            loss = _compute_loss(outputs, copy.deepcopy(prepared.targets), loss_fn)
            if not torch.is_tensor(loss):
                raise RuntimeError(f"baseline freeze loss returned {type(loss)!r}")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        model.eval()
        trainable_module.eval()
        with torch.no_grad():
            proxy_metric_after_by_epoch.append(evaluate_epoch(epoch))
        model.train()
        trainable_module.train()
    return {
        "microprofile_time_sec": time.perf_counter() - started,
        "loss_before": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "batch_count": len(losses),
        "proxy_metric_after_by_epoch": proxy_metric_after_by_epoch,
    }


def _prepare_raw_batch_for_full_forward(
    model: torch.nn.Module,
    trainable_module: torch.nn.Module,
    samples: list[RawFrameTrainingSample],
    *,
    device: torch.device,
) -> _PreparedBatch:
    raw_tensors: list[torch.Tensor] = []
    original_sizes: list[tuple[int, int]] = []
    for sample in samples:
        rgb = cv2.cvtColor(sample.image_bgr, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(rgb)
        raw_tensors.append(tensor)
        original_sizes.append((int(sample.image_bgr.shape[0]), int(sample.image_bgr.shape[1])))

    if trainable_module is not model:
        model_inputs = _prepare_split_adapter_batch(model, samples, device=device)
    else:
        model_inputs = torch.stack([tensor.to(device) for tensor in raw_tensors], dim=0)

    model_input_size = _infer_model_input_size(model_inputs)
    input_tensor_shape = _infer_input_tensor_shape(model_inputs)
    input_resize_mode = _input_resize_mode_for_model(model)
    targets = [
        _target_to_training_dict(
            sample.target,
            frame_id=sample.frame_id,
            original_image_size=original_size,
            model_input_size=model_input_size,
            input_tensor_shape=input_tensor_shape,
            input_resize_mode=input_resize_mode,
            device=device,
        )
        for sample, original_size in zip(samples, original_sizes)
    ]
    return _PreparedBatch(
        model_inputs=model_inputs,
        raw_image_tensors=[tensor.to(device) for tensor in raw_tensors],
        targets=targets,
    )


def _forward_full_model(
    model: torch.nn.Module,
    trainable_module: torch.nn.Module,
    prepared: _PreparedBatch,
) -> Any:
    if trainable_module is not model:
        return trainable_module(prepared.model_inputs)
    try:
        return model(prepared.model_inputs)
    except TypeError:
        return model(prepared.raw_image_tensors)


def _compute_loss(
    outputs: Any,
    targets: list[dict[str, Any]],
    loss_fn: Callable[[Any, Any], torch.Tensor] | None,
) -> torch.Tensor:
    if loss_fn is not None:
        return loss_fn(outputs, targets)
    if isinstance(outputs, Mapping):
        losses = [value for value in outputs.values() if torch.is_tensor(value)]
        if losses:
            return sum(value.sum() for value in losses)
    return _default_detection_count_loss(outputs, targets)


def _default_detection_count_loss(outputs: Any, targets: list[dict[str, Any]]) -> torch.Tensor:
    output_tensor = _first_tensor(outputs)
    if output_tensor is None:
        raise RuntimeError("Unable to compute baseline freeze loss from non-tensor outputs")
    flattened = output_tensor.reshape((output_tensor.shape[0], -1)).mean(dim=1)
    target_counts = torch.tensor(
        [float(len(target.get("boxes", []))) for target in targets],
        dtype=flattened.dtype,
        device=flattened.device,
    )
    if target_counts.numel() != flattened.numel():
        target_counts = target_counts[: flattened.numel()]
    return torch.nn.functional.mse_loss(flattened, target_counts)


def _target_to_training_dict(
    target: Mapping[str, Any],
    *,
    frame_id: int,
    original_image_size: tuple[int, int],
    model_input_size: tuple[int, int],
    input_tensor_shape: list[int],
    input_resize_mode: str,
    device: torch.device,
) -> dict[str, Any]:
    boxes = torch.as_tensor(
        _target_value_or_empty(target, "boxes"),
        dtype=torch.float32,
        device=device,
    )
    if boxes.ndim == 1:
        boxes = boxes.reshape((-1, 4)) if boxes.numel() else boxes.reshape((0, 4))
    labels = torch.as_tensor(
        _target_value_or_empty(target, "labels"),
        dtype=torch.int64,
        device=device,
    )
    if labels.ndim == 0:
        labels = labels.reshape((1,))
    if labels.numel() != boxes.shape[0]:
        labels = labels[: boxes.shape[0]]
        if labels.numel() < boxes.shape[0]:
            labels = torch.cat(
                [
                    labels,
                    torch.ones(
                        (boxes.shape[0] - labels.numel(),),
                        dtype=torch.int64,
                        device=device,
                    ),
                ]
            )
    result: dict[str, Any] = {
        "boxes": boxes,
        "labels": labels,
        "image_id": torch.tensor([int(frame_id)], dtype=torch.int64, device=device),
        "label_coordinate_space": ORIGINAL_XYXY,
        "label_image_size": [int(original_image_size[0]), int(original_image_size[1])],
        "_split_meta": {
            "input_image_size": [int(original_image_size[0]), int(original_image_size[1])],
            "input_tensor_shape": input_tensor_shape,
            "input_resize_mode": str(input_resize_mode or "direct_resize"),
            "model_input_size": [int(model_input_size[0]), int(model_input_size[1])],
        },
    }
    if "scores" in target:
        result["scores"] = torch.as_tensor(
            _target_value_or_empty(target, "scores"),
            dtype=torch.float32,
            device=device,
        )
    return result


def _target_value_or_empty(target: Mapping[str, Any], key: str) -> Any:
    value = target.get(key)
    return [] if value is None else value


def _prepare_split_adapter_batch(
    model: torch.nn.Module,
    samples: list[RawFrameTrainingSample],
    *,
    device: torch.device,
) -> Any:
    prepared = [
        prepare_split_runtime_input(model, sample.image_bgr, device=device) for sample in samples
    ]
    return _collate_prepared_inputs(prepared)


def _collate_prepared_inputs(prepared: list[Any]) -> Any:
    if not prepared:
        raise RuntimeError("baseline freeze training batch is empty")
    if all(torch.is_tensor(item) for item in prepared):
        tensors = [item for item in prepared if torch.is_tensor(item)]
        if all(tensor.ndim >= 4 and int(tensor.shape[0]) == 1 for tensor in tensors):
            return torch.cat(tensors, dim=0)
        return torch.stack(tensors, dim=0)
    if (
        all(isinstance(item, (list, tuple)) for item in prepared)
        and all(len(item) == 1 and torch.is_tensor(item[0]) for item in prepared)
    ):
        return [item[0] for item in prepared]
    if len(prepared) == 1:
        return prepared[0]
    raise RuntimeError(
        "Unable to collate baseline freeze preprocessed inputs: "
        f"types={[type(item).__name__ for item in prepared]}"
    )


def _input_resize_mode_for_model(model: torch.nn.Module) -> str:
    return str(get_split_runtime_input_resize_mode(model) or "direct_resize")


def _infer_model_input_size(value: Any) -> tuple[int, int]:
    shape = _infer_input_tensor_shape(value)
    if len(shape) < 2:
        raise RuntimeError("baseline freeze model input is missing image dimensions")
    return int(shape[-2]), int(shape[-1])


def _infer_input_tensor_shape(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(dim) for dim in value.shape]
    if isinstance(value, Mapping):
        for item in value.values():
            shape = _infer_input_tensor_shape(item)
            if shape:
                return shape
    if isinstance(value, (list, tuple)):
        for item in value:
            shape = _infer_input_tensor_shape(item)
            if shape:
                return shape
    return []


def _first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _batches(samples: list[RawFrameTrainingSample], batch_size: int):
    for index in range(0, len(samples), max(1, int(batch_size))):
        yield samples[index : index + max(1, int(batch_size))]


def _build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    *,
    learning_rate: float,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    params = [parameter for parameter in parameters if bool(parameter.requires_grad)]
    if not params:
        raise RuntimeError("no trainable parameters available")
    name = str(optimizer_name or "adam").strip().lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=float(learning_rate), weight_decay=float(weight_decay))
    if name == "sgd":
        return torch.optim.SGD(params, lr=float(learning_rate), weight_decay=float(weight_decay))
    return torch.optim.Adam(params, lr=float(learning_rate), weight_decay=float(weight_decay))


def _load_optional_base_model_update(
    model: torch.nn.Module,
    *,
    workspace_path: Path,
    manifest: Mapping[str, Any],
    device: torch.device,
) -> None:
    payload_bytes = b""
    update_path = str(manifest.get("base_model_update_path", "") or "")
    if update_path:
        path = workspace_path / update_path
        if path.exists():
            payload_bytes = path.read_bytes()
    if not payload_bytes:
        encoded = str(manifest.get("base_model_update_model_data", "") or "")
        if encoded:
            payload_bytes = base64.b64decode(encoded)
    if not payload_bytes:
        return
    payload = require_state_dict_delta_payload(
        torch.load(io.BytesIO(payload_bytes), map_location=device, weights_only=False)
    )
    state_dict = dict(payload["state_dict"])
    model.load_state_dict(state_dict, strict=False)
    logger.info(
        "[BaselineTraining] loaded base model update: state_keys={}",
        len(state_dict),
    )


def _trainable_param_ratio(training_cfg: Mapping[str, Any]) -> float:
    value = training_cfg.get("trainable_param_ratio", 0.3)
    try:
        ratio = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("trainable_param_ratio must be numeric") from exc
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError("trainable_param_ratio must be in (0, 1]")
    return ratio


def _serializable_freeze_summary(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in summary.items()
        if key != "selected_trainable_parameters"
    }
