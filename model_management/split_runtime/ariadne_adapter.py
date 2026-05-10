from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import torch
from ariadne import SplitSpec, prepare_split, prepare_split_replay

ARIADNE_RUNTIME_ADAPTER_VERSION = "plank-road-ariadne-runtime-v3"
DEFAULT_SPLIT_MODE = "generated_eager"


@dataclass(frozen=True)
class SplitRuntimeConfig:
    boundary: str
    dynamic_batch: tuple[int, int] = (2, 64)
    trace_batch_size: int = 2
    mode: Literal["generated_eager", "compiled"] = DEFAULT_SPLIT_MODE
    trainable: bool = True


def make_split_spec(config: SplitRuntimeConfig) -> SplitSpec:
    return SplitSpec(
        boundary=config.boundary,
        batch_symbol="B",
        dynamic_batch=config.dynamic_batch,
        trainable=config.trainable,
        trace_batch_mode="batch_gt1",
    )


def _require_batch_gt1(example_batch: torch.Tensor) -> None:
    if int(example_batch.shape[0]) <= 1:
        raise ValueError("Ariadne batch_gt1 tracing requires example_batch batch size > 1.")


def build_split_runtime(model: Any, example_batch: torch.Tensor, config: SplitRuntimeConfig) -> Any:
    _require_batch_gt1(example_batch)
    return prepare_split(
        model,
        example_inputs=(example_batch,),
        split=make_split_spec(config),
        mode=config.mode,
    )


def build_replay_runtime(model: Any, example_batch: torch.Tensor, config: SplitRuntimeConfig) -> Any:
    _require_batch_gt1(example_batch)
    spec = make_split_spec(replace(config, trainable=False))
    return prepare_split_replay(
        model,
        example_inputs=(example_batch,),
        split=spec,
        mode=config.mode,
        validation="strict",
        materialize_boundary=True,
    )


def maybe_warmup_runtime(runtime: Any, batch: torch.Tensor) -> None:
    run_prefix = getattr(runtime, "run_prefix", None)
    run_suffix = getattr(runtime, "run_suffix", None)
    if not callable(run_prefix) or not callable(run_suffix):
        return
    with torch.inference_mode():
        run_suffix(run_prefix(batch))


def get_split_runtime_metadata(runtime: Any) -> dict[str, Any]:
    candidate = getattr(runtime, "candidate", None)
    return {
        "actual_split_id": getattr(runtime, "split_id", None),
        "graph_signature": getattr(runtime, "graph_signature", None),
        "ariadne_mode": getattr(runtime, "mode", None),
        "boundary_after": getattr(candidate, "boundary_after", None),
        "boundary_nodes": list(getattr(candidate, "boundary_nodes", ()) or ()),
        "prefix_nodes": list(getattr(candidate, "prefix_nodes", ()) or ()),
        "suffix_nodes": list(getattr(candidate, "suffix_nodes", ()) or ()),
        "boundary_bytes": int(getattr(getattr(candidate, "cost", None), "boundary_bytes", 0) or 0),
        "trainable_suffix": bool(getattr(candidate, "trainable_suffix", False)),
    }


__all__ = [
    "ARIADNE_RUNTIME_ADAPTER_VERSION",
    "DEFAULT_SPLIT_MODE",
    "SplitRuntimeConfig",
    "build_replay_runtime",
    "build_split_runtime",
    "get_split_runtime_metadata",
    "make_split_spec",
    "maybe_warmup_runtime",
]
