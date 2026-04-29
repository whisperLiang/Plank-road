from __future__ import annotations

import time
from typing import Any, Callable

import torch

from .ariadne_runtime import BoundaryPayload, SplitRuntime
from .errors import (
    BatchPrefixError,
    BatchSuffixReplayError,
    BoundaryPayloadValidationError,
    MissingLossFunctionError,
    SplitTailTrainingError,
    boundary_tensor_shapes,
    error_context,
    input_shapes,
)


def _boundary_labels(boundary: BoundaryPayload) -> list[str]:
    return list(getattr(boundary, "tensors", {}).keys())


def _runtime_boundary_schema(runtime: SplitRuntime) -> dict[str, Any]:
    return dict(getattr(getattr(runtime, "candidate", None), "boundary_schema", {}) or {})


def _runtime_graph_signature(runtime: SplitRuntime, fallback: str) -> str:
    signature = getattr(runtime, "graph_signature", None)
    return str(signature if signature is not None else fallback)


def _tensor_matches_spec(tensor: torch.Tensor, tensor_spec: Any) -> bool:
    if str(tensor.dtype) != str(getattr(tensor_spec, "dtype", tensor.dtype)):
        return False
    symbolic_shape = tuple(getattr(tensor_spec, "symbolic_shape", ()) or ())
    if not symbolic_shape:
        return tensor.ndim == 0
    if tensor.ndim != len(symbolic_shape):
        return False
    for actual_dim, expected_dim in zip(tensor.shape[1:], symbolic_shape[1:]):
        try:
            if int(actual_dim) != int(expected_dim):
                return False
        except (TypeError, ValueError):
            continue
    return True


def _align_boundary_payload_schema(
    runtime: SplitRuntime,
    boundary: BoundaryPayload,
) -> BoundaryPayload:
    schema = _runtime_boundary_schema(runtime)
    if not schema:
        return boundary
    runtime_split_id = getattr(runtime, "split_id", None)
    if runtime_split_id is not None and str(boundary.split_id) != str(runtime_split_id):
        return boundary
    tensors = dict(getattr(boundary, "tensors", {}) or {})
    if list(tensors.keys()) == list(schema.keys()) and (
        str(boundary.graph_signature) == _runtime_graph_signature(runtime, boundary.graph_signature)
    ):
        return boundary
    if len(tensors) != len(schema):
        return boundary

    used_labels: set[str] = set()
    remapped_tensors: dict[str, torch.Tensor] = {}
    remapped_requires_grad: dict[str, bool] = {}
    for target_label, tensor_spec in schema.items():
        source_label = target_label if target_label in tensors else None
        if source_label is None:
            for candidate_label, candidate_tensor in tensors.items():
                if candidate_label in used_labels or not isinstance(candidate_tensor, torch.Tensor):
                    continue
                if _tensor_matches_spec(candidate_tensor, tensor_spec):
                    source_label = candidate_label
                    break
        if source_label is None:
            return boundary
        source_tensor = tensors[source_label]
        if not isinstance(source_tensor, torch.Tensor) or not _tensor_matches_spec(
            source_tensor,
            tensor_spec,
        ):
            return boundary
        used_labels.add(source_label)
        remapped_tensors[str(target_label)] = source_tensor
        remapped_requires_grad[str(target_label)] = bool(
            getattr(boundary, "requires_grad", {}).get(source_label, source_tensor.requires_grad)
        )

    return BoundaryPayload(
        split_id=str(runtime_split_id or boundary.split_id),
        graph_signature=_runtime_graph_signature(runtime, boundary.graph_signature),
        batch_size=boundary.batch_size,
        tensors=remapped_tensors,
        schema=schema,
        requires_grad=remapped_requires_grad,
        weight_version=boundary.weight_version,
        passthrough_inputs=dict(boundary.passthrough_inputs or {}),
    )


def _move_value_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_value_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_value_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_value_to_device(item, device) for item in value]
    return value


def _value_needs_device_move(value: Any, device: torch.device) -> bool:
    if isinstance(value, torch.Tensor):
        return value.device != device
    if isinstance(value, dict):
        return any(_value_needs_device_move(item, device) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_value_needs_device_move(item, device) for item in value)
    return False


def _runtime_boundary_device(runtime: SplitRuntime, boundary: BoundaryPayload) -> torch.device | None:
    schema = _runtime_boundary_schema(runtime)
    for label in getattr(boundary, "tensors", {}) or {}:
        tensor_spec = schema.get(label)
        device_type = getattr(tensor_spec, "device_type", None)
        if device_type:
            return torch.device(str(device_type))
    root_module = getattr(getattr(runtime, "trace_plan", None), "root_module", None)
    if root_module is not None:
        for parameter in root_module.parameters():
            return parameter.device
    return None


def _align_boundary_payload_device(
    runtime: SplitRuntime,
    boundary: BoundaryPayload,
) -> BoundaryPayload:
    target_device = _runtime_boundary_device(runtime, boundary)
    if target_device is None:
        return boundary
    tensors = getattr(boundary, "tensors", {}) or {}
    passthrough_inputs = getattr(boundary, "passthrough_inputs", {}) or {}
    if all(
        not isinstance(tensor, torch.Tensor) or tensor.device == target_device
        for tensor in tensors.values()
    ) and not _value_needs_device_move(passthrough_inputs, target_device):
        return boundary
    return BoundaryPayload(
        split_id=boundary.split_id,
        graph_signature=boundary.graph_signature,
        batch_size=boundary.batch_size,
        tensors={
            label: tensor.to(target_device) if isinstance(tensor, torch.Tensor) else tensor
            for label, tensor in tensors.items()
        },
        schema=boundary.schema,
        requires_grad=boundary.requires_grad,
        weight_version=boundary.weight_version,
        passthrough_inputs=_move_value_to_device(passthrough_inputs, target_device),
    )


def _boundary_context(
    boundary: BoundaryPayload,
    *,
    model_name: str | None,
    model_family: str | None,
) -> str:
    return error_context(
        model_name=model_name,
        model_family=model_family,
        split_id=getattr(boundary, "split_id", None),
        graph_signature=getattr(boundary, "graph_signature", None),
        batch_size=getattr(boundary, "batch_size", None),
        boundary_labels=_boundary_labels(boundary),
        shapes=boundary_tensor_shapes(boundary),
    )


def validate_boundary_payload(
    runtime: SplitRuntime,
    boundary: BoundaryPayload,
    *,
    model_name: str | None = None,
    model_family: str | None = None,
) -> None:
    prepare_validated_boundary_payload(
        runtime,
        boundary,
        model_name=model_name,
        model_family=model_family,
    )


def prepare_validated_boundary_payload(
    runtime: SplitRuntime,
    boundary: BoundaryPayload,
    *,
    model_name: str | None = None,
    model_family: str | None = None,
) -> BoundaryPayload:
    boundary = _align_boundary_payload_schema(runtime, boundary)
    boundary = _align_boundary_payload_device(runtime, boundary)
    try:
        runtime.validate_boundary(boundary)
    except BaseException as exc:  # noqa: BLE001 - wrap Ariadne validation errors with context; allows KeyboardInterrupt to propagate
        context = _boundary_context(
            boundary,
            model_name=model_name,
            model_family=model_family,
        )
        raise BoundaryPayloadValidationError(
            f"Boundary payload validation failed ({context}): {exc}"
        ) from exc
    return boundary


def run_batch_prefix(
    runtime: SplitRuntime,
    *inputs: Any,
    model_name: str | None = None,
    model_family: str | None = None,
) -> BoundaryPayload:
    try:
        return runtime.run_prefix(*inputs)
    except Exception as exc:  # noqa: BLE001
        batch_size = None
        for value in inputs:
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                batch_size = int(value.shape[0])
                break
        context = error_context(
            model_name=model_name,
            model_family=model_family,
            split_id=getattr(runtime, "split_id", None),
            graph_signature=getattr(runtime, "graph_signature", None),
            batch_size=batch_size,
            shapes=input_shapes(tuple(inputs)),
        )
        raise BatchPrefixError(
            f"Batch prefix execution failed ({context}): {exc}"
        ) from exc


def run_batch_suffix(
    runtime: SplitRuntime,
    boundary: BoundaryPayload,
    *,
    model_name: str | None = None,
    model_family: str | None = None,
) -> Any:
    boundary = prepare_validated_boundary_payload(
        runtime,
        boundary,
        model_name=model_name,
        model_family=model_family,
    )
    try:
        return runtime.run_suffix(boundary)
    except Exception as exc:  # noqa: BLE001
        context = _boundary_context(
            boundary,
            model_name=model_name,
            model_family=model_family,
        )
        raise BatchSuffixReplayError(
            f"Batch suffix replay failed ({context}): {exc}"
        ) from exc


def train_batch_suffix(
    runtime: SplitRuntime,
    boundary: BoundaryPayload,
    targets: Any,
    *,
    loss_fn: Callable[[Any, Any], torch.Tensor] | None,
    optimizer: torch.optim.Optimizer | None = None,
    model_name: str | None = None,
    model_family: str | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
    if loss_fn is None:
        context = error_context(
            model_name=model_name,
            model_family=model_family,
            split_id=getattr(boundary, "split_id", None),
            batch_size=getattr(boundary, "batch_size", None),
        )
        raise MissingLossFunctionError(
            f"Split-tail training requires an adapter loss function ({context})."
        )
    boundary = prepare_validated_boundary_payload(
        runtime,
        boundary,
        model_name=model_name,
        model_family=model_family,
    )
    try:
        loss, gradients = runtime.train_suffix(
            boundary,
            targets,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
    except Exception as exc:  # noqa: BLE001
        context = _boundary_context(
            boundary,
            model_name=model_name,
            model_family=model_family,
        )
        raise SplitTailTrainingError(
            f"Split-tail training failed ({context}): {exc}"
        ) from exc
    if not torch.isfinite(loss.detach()):
        context = error_context(
            model_name=model_name,
            model_family=model_family,
            split_id=getattr(boundary, "split_id", None),
            batch_size=getattr(boundary, "batch_size", None),
        )
        raise SplitTailTrainingError(
            f"Split-tail training produced a non-finite loss ({context})."
        )
    return loss, dict(gradients or {})


def train_batch_suffix_fast(
    runtime: SplitRuntime,
    boundary: BoundaryPayload,
    targets: Any,
    *,
    loss_fn: Callable[[Any, Any], torch.Tensor] | None,
    optimizer: torch.optim.Optimizer | None = None,
    model_name: str | None = None,
    model_family: str | None = None,
    profile: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
    if loss_fn is None:
        context = error_context(
            model_name=model_name,
            model_family=model_family,
            split_id=getattr(boundary, "split_id", None),
            batch_size=getattr(boundary, "batch_size", None),
        )
        raise MissingLossFunctionError(
            f"Split-tail training requires an adapter loss function ({context})."
        )
    try:
        if optimizer is not None:
            optimizer.zero_grad()
        forward_backward_started = time.perf_counter()
        outputs = runtime.run_suffix(boundary)
        loss = loss_fn(outputs, targets)
        if not isinstance(loss, torch.Tensor):
            raise TypeError(f"Split-tail loss function returned {type(loss)!r}.")
        if loss.requires_grad:
            loss.backward()
        if profile is not None:
            profile["suffix_forward_backward_time"] = (
                float(profile.get("suffix_forward_backward_time", 0.0))
                + time.perf_counter()
                - forward_backward_started
            )
        step_started = time.perf_counter()
        if optimizer is not None:
            optimizer.step()
        if profile is not None:
            profile["optimizer_step_time"] = (
                float(profile.get("optimizer_step_time", 0.0))
                + time.perf_counter()
                - step_started
            )
    except Exception as exc:  # noqa: BLE001
        context = _boundary_context(
            boundary,
            model_name=model_name,
            model_family=model_family,
        )
        raise SplitTailTrainingError(
            f"Fast split-tail training failed ({context}): {exc}"
        ) from exc
    if not torch.isfinite(loss.detach()):
        context = error_context(
            model_name=model_name,
            model_family=model_family,
            split_id=getattr(boundary, "split_id", None),
            batch_size=getattr(boundary, "batch_size", None),
        )
        raise SplitTailTrainingError(
            f"Split-tail training produced a non-finite loss ({context})."
        )
    return loss, {}


__all__ = [
    "prepare_validated_boundary_payload",
    "run_batch_prefix",
    "run_batch_suffix",
    "train_batch_suffix",
    "train_batch_suffix_fast",
    "validate_boundary_payload",
]
