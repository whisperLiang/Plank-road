from __future__ import annotations

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
    validate_boundary_payload(
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
    validate_boundary_payload(
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


__all__ = [
    "run_batch_prefix",
    "run_batch_suffix",
    "train_batch_suffix",
    "validate_boundary_payload",
]
