from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from cloud.baselines.ekya_style_cloud_scheduling.config import RetrainingConfig
from cloud.training.parameter_freeze import (
    RawFrameTrainingSample,
    apply_parameter_ratio_freeze,
    selected_trainable_parameters,
    unwrap_trainable_module,
)
from cloud.training.strategies.baseline_freeze import (
    build_baseline_freeze_loss,
    run_parameter_ratio_freeze_training,
)


@dataclass(frozen=True)
class TrainingComponents:
    model: torch.nn.Module
    trainable_module: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: Any
    device: torch.device
    trainable_summary: dict[str, Any]


def resolve_training_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_base_state_dict(
    model: torch.nn.Module,
    base_state_dict: Mapping[str, Any],
) -> None:
    if not isinstance(base_state_dict, Mapping) or not base_state_dict:
        raise RuntimeError("Ekya training requires base model weights")
    model.load_state_dict(dict(base_state_dict), strict=False)


def build_training_components(
    *,
    model: torch.nn.Module,
    config: RetrainingConfig,
    learning_rate: float,
) -> TrainingComponents:
    device = resolve_training_device()
    model.to(device)
    trainable_module = unwrap_trainable_module(model, model_name="rfdetr_nano")
    trainable_module.to(device)
    train_mode = str(config.train_mode or "full").strip().lower()
    if train_mode == "full":
        selected = _enable_full_training(trainable_module)
        summary = {
            "train_mode": "full",
            "trainable_tensors": len(selected),
            "trainable_params": sum(parameter.numel() for parameter in selected),
        }
    elif train_mode == "freeze":
        ratio = config.trainable_param_ratio
        if ratio is None:
            raise ValueError("train_mode=freeze requires trainable_param_ratio")
        freeze_summary = apply_parameter_ratio_freeze(trainable_module, float(ratio))
        selected = [
            parameter for _name, parameter in selected_trainable_parameters(freeze_summary)
        ]
        summary = {"train_mode": "freeze", **_serializable_summary(freeze_summary)}
    else:
        raise ValueError(f"unsupported Ekya train_mode: {config.train_mode!r}")
    optimizer = _build_optimizer(
        selected,
        learning_rate=float(learning_rate),
        optimizer_name=config.optimizer_name,
        weight_decay=float(config.weight_decay),
    )
    return TrainingComponents(
        model=model,
        trainable_module=trainable_module,
        optimizer=optimizer,
        loss_fn=build_baseline_freeze_loss(model),
        device=device,
        trainable_summary=summary,
    )


def run_one_training_epoch(
    *,
    components: TrainingComponents,
    samples: Iterable[RawFrameTrainingSample],
    batch_size: int,
) -> tuple[float | None, dict[str, Any]]:
    metrics = run_parameter_ratio_freeze_training(
        model=components.model,
        trainable_module=components.trainable_module,
        samples=list(samples),
        batch_size=max(1, int(batch_size)),
        epochs=1,
        device=components.device,
        loss_fn=components.loss_fn,
        optimizer=components.optimizer,
        log_epochs=False,
    )
    epoch_losses = list(metrics.get("epoch_losses") or [])
    loss = float(epoch_losses[-1]) if epoch_losses else metrics.get("final_loss")
    return (None if loss is None else float(loss)), metrics


def cpu_state_dict(model: torch.nn.Module) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for key, value in model.state_dict().items():
        if torch.is_tensor(value):
            state[key] = value.detach().cpu().clone()
        else:
            state[key] = value
    if not state:
        raise RuntimeError("Ekya training produced no model weights")
    return state


def assert_non_empty_checkpoint_state(path: str) -> bool:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        return False
    state = checkpoint.get("state_dict")
    return isinstance(state, Mapping) and bool(state)


def _enable_full_training(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    selected: list[torch.nn.Parameter] = []
    for parameter in module.parameters():
        if isinstance(parameter, torch.nn.Parameter) and parameter.dtype.is_floating_point:
            parameter.requires_grad_(True)
            parameter.grad = None
            selected.append(parameter)
    if not selected:
        raise RuntimeError("Ekya full training found no floating-point trainable parameters")
    return selected


def _build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    *,
    learning_rate: float,
    optimizer_name: str,
    weight_decay: float,
) -> torch.optim.Optimizer:
    params = [parameter for parameter in parameters if bool(parameter.requires_grad)]
    if not params:
        raise RuntimeError("Ekya training has no trainable parameters")
    name = str(optimizer_name or "adamw").strip().lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=float(learning_rate), weight_decay=float(weight_decay))
    if name == "adam":
        return torch.optim.Adam(params, lr=float(learning_rate), weight_decay=float(weight_decay))
    if name == "sgd":
        return torch.optim.SGD(params, lr=float(learning_rate), weight_decay=float(weight_decay))
    raise ValueError(f"unsupported Ekya optimizer_name: {optimizer_name!r}")


def _serializable_summary(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        str(key): value
        for key, value in summary.items()
        if key != "selected_trainable_parameters"
    }
