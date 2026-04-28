from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ariadne import BoundaryPayload, SplitRuntime, SplitSpec, prepare_split

ARIADNE_RUNTIME_ADAPTER_VERSION = "plank-road-ariadne-runtime-v1"
DEFAULT_SPLIT_MODE = "generated_eager"


def normalize_example_inputs(example_inputs: Any) -> tuple[Any, ...]:
    if isinstance(example_inputs, tuple):
        return example_inputs
    if isinstance(example_inputs, list):
        return tuple(example_inputs)
    return (example_inputs,)


def make_split_spec(
    boundary: str,
    *,
    batch_symbol: str = "B",
    dynamic_batch: tuple[int, int] | None = (2, 64),
    trainable: bool = True,
    trace_batch_mode: str = "batch_gt1",
    model_family: str | None = None,
) -> SplitSpec:
    """Build an Ariadne SplitSpec.

    Ariadne 0.1.0 does not expose ``model_family`` on SplitSpec; Plank-road
    carries that value in cache keys and logs instead of mutating Ariadne's API.
    """

    _ = model_family
    return SplitSpec(
        boundary=str(boundary),
        batch_symbol=batch_symbol,
        dynamic_batch=dynamic_batch,
        trainable=bool(trainable),
        trace_batch_mode=trace_batch_mode,
    )


def prepare_split_runtime(
    model,
    example_inputs: Sequence[Any] | Any,
    split_spec: SplitSpec | str,
    mode: str = DEFAULT_SPLIT_MODE,
) -> SplitRuntime:
    """Prepare an Ariadne SplitRuntime without adding a second executor."""

    inputs = normalize_example_inputs(example_inputs)
    split = SplitSpec(boundary=split_spec) if isinstance(split_spec, str) else split_spec
    return prepare_split(
        model,
        example_inputs=inputs,
        split=split,
        mode=mode,
    )


__all__ = [
    "ARIADNE_RUNTIME_ADAPTER_VERSION",
    "BoundaryPayload",
    "DEFAULT_SPLIT_MODE",
    "SplitRuntime",
    "SplitSpec",
    "make_split_spec",
    "normalize_example_inputs",
    "prepare_split_runtime",
]
