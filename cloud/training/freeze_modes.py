from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F

from cloud.training.adapters import train_split_suffix_batch
from model_management.universal_model_split import _suffix_parameter_names


@dataclass(frozen=True)
class RawFrameTrainingSample:
    frame_id: int
    image_bgr: np.ndarray
    target: Mapping[str, Any]


def configure_fixed_prefix_training(
    split_model: torch.nn.Module,
    runtime: Any,
) -> tuple[tuple[str, ...], list[torch.nn.Parameter]]:
    torchlens_runtime = (
        runtime._ensure_runtime()
        if callable(getattr(runtime, "_ensure_runtime", None))
        else runtime
    )
    suffix_names = tuple(_suffix_parameter_names(runtime))
    suffix_name_set = set(suffix_names)

    split_model.eval()
    for parameter in split_model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None

    suffix_segment = getattr(torchlens_runtime, "suffix_segment", None)
    if isinstance(suffix_segment, torch.nn.Module):
        suffix_segment.train()

    for segment_name in ("prefix_segment", "training_prefix_segment"):
        prefix_segment = getattr(torchlens_runtime, segment_name, None)
        if not isinstance(prefix_segment, torch.nn.Module):
            continue
        prefix_segment.eval()
        for parameter in prefix_segment.parameters(recurse=True):
            parameter.requires_grad_(False)
            parameter.grad = None

    suffix_params: list[torch.nn.Parameter] = []
    named_parameters = dict(split_model.named_parameters())
    for name, parameter in named_parameters.items():
        if name in suffix_name_set:
            parameter.requires_grad_(True)
            suffix_params.append(parameter)

    missing = sorted(suffix_name_set - set(named_parameters.keys()))
    if missing:
        raise RuntimeError(
            "Suffix trainable parameters missing from split model: " + ", ".join(missing)
        )
    if not suffix_params:
        raise RuntimeError("No suffix parameters were selected for freeze training")
    return suffix_names, suffix_params


def run_freeze_training(
    *,
    model: torch.nn.Module,
    runtime: Any,
    samples: Iterable[RawFrameTrainingSample],
    batch_size: int,
    epochs: int,
    device: torch.device,
    loss_fn: Callable[[Any, Any], torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    sample_list = list(samples)
    configure_fixed_prefix_training(model, runtime)
    losses: list[float] = []
    started = time.perf_counter()
    for _epoch in range(int(epochs)):
        for batch in _batches(sample_list, max(1, int(batch_size))):
            inputs, targets = _prepare_raw_batch(batch, device=device)
            _set_runtime_prefix_module_state(runtime)
            with torch.no_grad():
                boundary = runtime.run_prefix(inputs)
            _set_runtime_suffix_module_state(runtime)
            loss = train_split_suffix_batch(
                runtime,
                boundary,
                copy.deepcopy(targets),
                loss_fn,
                optimizer,
            )
            if not torch.is_tensor(loss):
                raise RuntimeError(f"freeze train_suffix returned {type(loss)!r}")
            losses.append(float(loss.detach().cpu().item()))
    return {
        "suffix_train_time_sec": time.perf_counter() - started,
        "feature_rebuild_time_sec": 0.0,
        "final_loss": losses[-1] if losses else None,
        "batch_count": len(losses),
    }


def decode_training_sample(
    *,
    frame_id: int,
    raw_frame: bytes,
    target: Mapping[str, Any],
) -> RawFrameTrainingSample:
    array = np.frombuffer(raw_frame, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"unable to decode baseline frame {frame_id}")
    return RawFrameTrainingSample(frame_id=int(frame_id), image_bgr=image, target=dict(target))


def build_optimizer(
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


def _prepare_raw_batch(
    samples: list[RawFrameTrainingSample],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    tensors = []
    targets = []
    for sample in samples:
        rgb = cv2.cvtColor(sample.image_bgr, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(rgb).to(device)
        tensors.append(tensor)
        target = _target_to_device(_target_to_tensors(sample.target), device)
        target["image_id"] = torch.tensor([int(sample.frame_id)], dtype=torch.int64, device=device)
        targets.append(target)
    return torch.stack(tensors, dim=0), targets


def _target_to_tensors(target: Mapping[str, Any]) -> dict[str, Any]:
    boxes = torch.as_tensor(target.get("boxes", []) or [], dtype=torch.float32)
    if boxes.ndim == 1:
        boxes = boxes.reshape((-1, 4)) if boxes.numel() else boxes.reshape((0, 4))
    labels = torch.as_tensor(target.get("labels", []) or [], dtype=torch.int64)
    if labels.ndim == 0:
        labels = labels.reshape((1,))
    if labels.numel() != boxes.shape[0]:
        labels = labels[: boxes.shape[0]]
        if labels.numel() < boxes.shape[0]:
            labels = torch.cat(
                [labels, torch.ones((boxes.shape[0] - labels.numel(),), dtype=torch.int64)]
            )
    result: dict[str, Any] = {"boxes": boxes, "labels": labels}
    if "scores" in target:
        result["scores"] = torch.as_tensor(target.get("scores") or [], dtype=torch.float32)
    return result


def _target_to_device(target: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in target.items()
    }


def _batches(samples: list[RawFrameTrainingSample], batch_size: int):
    for index in range(0, len(samples), max(1, int(batch_size))):
        yield samples[index : index + max(1, int(batch_size))]


def _set_runtime_prefix_module_state(runtime: Any) -> None:
    torchlens_runtime = (
        runtime._ensure_runtime()
        if callable(getattr(runtime, "_ensure_runtime", None))
        else runtime
    )
    for segment_name in ("prefix_segment", "training_prefix_segment"):
        prefix_segment = getattr(torchlens_runtime, segment_name, None)
        if isinstance(prefix_segment, torch.nn.Module):
            prefix_segment.eval()


def _set_runtime_suffix_module_state(runtime: Any) -> None:
    torchlens_runtime = (
        runtime._ensure_runtime()
        if callable(getattr(runtime, "_ensure_runtime", None))
        else runtime
    )
    suffix_segment = getattr(torchlens_runtime, "suffix_segment", None)
    if isinstance(suffix_segment, torch.nn.Module):
        suffix_segment.train()
