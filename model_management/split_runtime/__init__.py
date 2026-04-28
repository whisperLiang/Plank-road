from __future__ import annotations

from typing import Any

import torch

from .ariadne_runtime import (
    ARIADNE_RUNTIME_ADAPTER_VERSION,
    BoundaryPayload,
    SplitRuntime,
    SplitSpec,
    make_split_spec,
    prepare_split_runtime,
)
from .detection_adapters import (
    DetectionSplitAdapter,
    PlankDetectionSplitAdapter,
    select_detection_adapter,
)
from .errors import (
    BatchPrefixError,
    BatchSuffixReplayError,
    BoundaryPayloadValidationError,
    InvalidOutputStructureError,
    MissingLossFunctionError,
    SplitRuntimeError,
    SplitTailTrainingError,
    UnsupportedModelAdapterError,
)
from .runtime_cache import RuntimeCache, RuntimeCacheKey, make_runtime_cache_key
from .template import (
    FIXED_SPLIT_RUNTIME_TEMPLATE_CACHE_VERSION,
    FixedSplitRuntimeTemplate,
    FixedSplitRuntimeTemplateCache,
    FixedSplitRuntimeTemplateKey,
    FixedSplitRuntimeTemplateLookup,
    bind_request_runtime_from_template,
    fixed_split_runtime_template_key,
    get_fixed_split_runtime_template_cache,
)
from .validators import (
    run_batch_prefix,
    run_batch_suffix,
    train_batch_suffix,
    validate_boundary_payload,
)


def _flatten_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        tensors: list[torch.Tensor] = []
        for item in value.values():
            tensors.extend(_flatten_tensors(item))
        return tensors
    if isinstance(value, (list, tuple)):
        tensors = []
        for item in value:
            tensors.extend(_flatten_tensors(item))
        return tensors
    return []


def reduce_output_to_loss(outputs: Any, targets: Any = None) -> torch.Tensor:
    del targets
    tensors = [tensor for tensor in _flatten_tensors(outputs) if tensor.is_floating_point()]
    if not tensors:
        raise RuntimeError("Could not reduce structured output to a differentiable scalar.")
    loss = tensors[0].sum() * 0.0
    pieces = 0
    for tensor in tensors:
        if tensor.numel() == 0:
            continue
        finite = tensor[torch.isfinite(tensor)]
        if finite.numel() == 0:
            continue
        loss = loss + finite.float().mean()
        pieces += 1
    return loss / max(1, pieces)


def compare_outputs(
    expected: Any,
    replayed: Any,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> tuple[bool, float]:
    expected_tensors = _flatten_tensors(expected)
    replayed_tensors = _flatten_tensors(replayed)
    if len(expected_tensors) != len(replayed_tensors):
        return False, float("inf")
    max_diff = 0.0
    for lhs, rhs in zip(expected_tensors, replayed_tensors, strict=True):
        if tuple(lhs.shape) != tuple(rhs.shape):
            return False, float("inf")
        lhs_cpu = lhs.detach().cpu()
        rhs_cpu = rhs.detach().cpu()
        if lhs_cpu.numel() == 0:
            continue
        diff = float((lhs_cpu - rhs_cpu).abs().max().item())
        max_diff = max(max_diff, diff)
        if not torch.allclose(lhs_cpu, rhs_cpu, atol=atol, rtol=rtol):
            return False, max_diff
    return True, max_diff


__all__ = [
    "ARIADNE_RUNTIME_ADAPTER_VERSION",
    "BatchPrefixError",
    "BatchSuffixReplayError",
    "BoundaryPayload",
    "BoundaryPayloadValidationError",
    "DetectionSplitAdapter",
    "FIXED_SPLIT_RUNTIME_TEMPLATE_CACHE_VERSION",
    "FixedSplitRuntimeTemplate",
    "FixedSplitRuntimeTemplateCache",
    "FixedSplitRuntimeTemplateKey",
    "FixedSplitRuntimeTemplateLookup",
    "InvalidOutputStructureError",
    "MissingLossFunctionError",
    "PlankDetectionSplitAdapter",
    "RuntimeCache",
    "RuntimeCacheKey",
    "SplitRuntime",
    "SplitRuntimeError",
    "SplitSpec",
    "SplitTailTrainingError",
    "UnsupportedModelAdapterError",
    "bind_request_runtime_from_template",
    "compare_outputs",
    "fixed_split_runtime_template_key",
    "get_fixed_split_runtime_template_cache",
    "make_runtime_cache_key",
    "make_split_spec",
    "prepare_split_runtime",
    "reduce_output_to_loss",
    "run_batch_prefix",
    "run_batch_suffix",
    "select_detection_adapter",
    "train_batch_suffix",
    "validate_boundary_payload",
]
