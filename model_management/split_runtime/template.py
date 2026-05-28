from __future__ import annotations

import threading
import time
from collections.abc import Callable, Hashable
from dataclasses import dataclass, replace
from typing import Any

import torch
from ariadne.codegen.segment_builder import build_segments
from ariadne.trace.interception import ConstantTensorArg
from loguru import logger

from .ariadne_runtime import SplitRuntime, SplitSpec
from .runtime_cache import RuntimeCacheKey, make_runtime_cache_key

FIXED_SPLIT_RUNTIME_TEMPLATE_CACHE_VERSION = 3


@dataclass(frozen=True)
class FixedSplitRuntimeTemplateKey(RuntimeCacheKey):
    version: int = FIXED_SPLIT_RUNTIME_TEMPLATE_CACHE_VERSION
    trace_batch_size: int | None = None
    validated_batch_max: int | None = None
    runtime_batch_validation_signature: str | None = None

    def as_dict(self) -> dict[str, object]:
        payload = super().as_dict()
        payload["version"] = int(self.version)
        if self.trace_batch_size is not None:
            payload["trace_batch_size"] = int(self.trace_batch_size)
        if self.validated_batch_max is not None:
            payload["validated_batch_max"] = int(self.validated_batch_max)
        if self.runtime_batch_validation_signature:
            payload["runtime_batch_validation_signature"] = (
                self.runtime_batch_validation_signature
            )
        return payload


def fixed_split_runtime_template_key(
    *,
    model_name: str,
    model_family: str,
    split_spec: SplitSpec,
    example_inputs: Any,
    graph_signature: str | None = None,
    split_plan_hash: str | None = None,
    trace_batch_size: int | None = None,
    validated_batch_max: int | None = None,
    runtime_batch_validation_signature: str | None = None,
    mode: str = "generated_eager",
) -> FixedSplitRuntimeTemplateKey:
    key = make_runtime_cache_key(
        model_name=model_name,
        model_family=model_family,
        split_spec=split_spec,
        example_inputs=example_inputs,
        graph_signature=graph_signature,
        split_plan_hash_value=split_plan_hash,
        mode=mode,
    )
    return FixedSplitRuntimeTemplateKey(
        **key.__dict__,
        trace_batch_size=None if trace_batch_size is None else int(trace_batch_size),
        validated_batch_max=(
            None if validated_batch_max is None else int(validated_batch_max)
        ),
        runtime_batch_validation_signature=runtime_batch_validation_signature,
    )


@dataclass(frozen=True)
class FixedSplitRuntimeTemplate:
    cache_key: FixedSplitRuntimeTemplateKey
    runtime: SplitRuntime
    split_spec: SplitSpec
    model_name: str
    model_family: str
    graph_signature: str | None
    symbolic_input_schema_hash: str
    split_plan_hash: str
    mode: str = "generated_eager"
    runtime_device: str = ""


@dataclass(frozen=True)
class FixedSplitRuntimeTemplateLookup:
    template: FixedSplitRuntimeTemplate
    cache_status: str
    wait_time_sec: float = 0.0
    cold_build_time_sec: float = 0.0


@dataclass
class _InflightTemplateBuild:
    event: threading.Event
    template: FixedSplitRuntimeTemplate | None = None
    error: BaseException | None = None


class FixedSplitRuntimeTemplateCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._templates: dict[Hashable, FixedSplitRuntimeTemplate] = {}
        self._inflight: dict[Hashable, _InflightTemplateBuild] = {}

    def get_or_create_lookup(
        self,
        cache_key: Hashable,
        builder: Callable[[], FixedSplitRuntimeTemplate],
    ) -> FixedSplitRuntimeTemplateLookup:
        with self._lock:
            cached = self._templates.get(cache_key)
            if cached is not None:
                logger.info(
                    "[FixedSplitCL] Ariadne runtime template cache hit (key={}).",
                    cache_key,
                )
                return FixedSplitRuntimeTemplateLookup(template=cached, cache_status="hit")

            inflight = self._inflight.get(cache_key)
            if inflight is None:
                inflight = _InflightTemplateBuild(event=threading.Event())
                self._inflight[cache_key] = inflight
                build_owner = True
                logger.info(
                    "[FixedSplitCL] Ariadne runtime template cache miss (key={}).",
                    cache_key,
                )
            else:
                build_owner = False
                logger.info(
                    "[FixedSplitCL] Waiting for Ariadne runtime template build (key={}).",
                    cache_key,
                )

        if not build_owner:
            wait_started = time.perf_counter()
            inflight.event.wait()
            wait_time = time.perf_counter() - wait_started
            if inflight.error is not None:
                raise inflight.error
            if inflight.template is None:
                raise RuntimeError(
                    "Ariadne runtime template build completed without a template "
                    f"(key={cache_key})."
                )
            return FixedSplitRuntimeTemplateLookup(
                template=inflight.template,
                cache_status="wait",
                wait_time_sec=wait_time,
            )

        started = time.perf_counter()
        try:
            template = builder()
        except BaseException as exc:
            with self._lock:
                inflight.error = exc
                self._inflight.pop(cache_key, None)
                inflight.event.set()
            raise

        elapsed = time.perf_counter() - started
        with self._lock:
            self._templates[cache_key] = template
            inflight.template = template
            self._inflight.pop(cache_key, None)
            inflight.event.set()
        logger.info(
            "[FixedSplitCL] Ariadne runtime template cold build completed in {:.3f}s (key={}).",
            elapsed,
            cache_key,
        )
        return FixedSplitRuntimeTemplateLookup(
            template=template,
            cache_status="miss",
            cold_build_time_sec=elapsed,
        )

    def get_or_create(
        self,
        cache_key: Hashable,
        builder: Callable[[], FixedSplitRuntimeTemplate],
    ) -> FixedSplitRuntimeTemplate:
        return self.get_or_create_lookup(cache_key, builder).template

    def clear(self) -> None:
        with self._lock:
            self._templates.clear()
            self._inflight.clear()


def bind_request_runtime_from_template(
    template: FixedSplitRuntimeTemplate,
    *,
    model: Any | None = None,
    device: str | None = None,
) -> SplitRuntime:
    """Bind a request-specific runtime from a cached template.
    
    Args:
        template: The cached template to bind.
        model: Reserved for future use (per-request model customization).
        device: Reserved for future use (device-specific binding).
    
    Returns:
        A split runtime bound to the request model.
    """
    _ = device  # The request model already carries the desired device.
    if model is None:
        return template.runtime
    return _rebind_runtime_to_model(template.runtime, model)


def _rebind_runtime_to_model(runtime: SplitRuntime, model: Any) -> SplitRuntime:
    trace_plan = _rebind_trace_plan_to_model(runtime.trace_plan, model)
    candidate = _rebind_candidate_device(getattr(runtime, "candidate", None), model)
    variants = tuple(
        _rebind_runtime_to_model(variant, model)
        for variant in tuple(getattr(runtime, "variants", ()) or ())
    )
    return SplitRuntime(
        trace_plan=trace_plan,
        split_spec=runtime.split_spec,
        candidate=candidate,
        segments=build_segments(trace_plan, candidate),
        mode=runtime.mode,
        variants=variants,
        batch_range=getattr(runtime, "batch_range", None),
    )


def _model_device(model: Any) -> Any:
    if model is None:
        return None
    for parameter in getattr(model, "parameters", lambda **_: ())():
        target_device = getattr(parameter, "device", None)
        if target_device is not None:
            return target_device
    for buffer in getattr(model, "buffers", lambda **_: ())():
        target_device = getattr(buffer, "device", None)
        if target_device is not None:
            return target_device
    return None


def _rebind_trace_plan_to_model(trace_plan: Any, model: Any) -> Any:
    target_device = _model_device(model)
    artifact = getattr(trace_plan, "runtime_artifact", None)
    if target_device is not None and artifact is not None:
        ops = tuple(
            replace(
                op,
                args_template=_move_template_tensors_to_device(
                    getattr(op, "args_template", None),
                    target_device,
                ),
                kwargs_template=_move_template_tensors_to_device(
                    getattr(op, "kwargs_template", None),
                    target_device,
                ),
                output_template=_move_template_tensors_to_device(
                    getattr(op, "output_template", None),
                    target_device,
                ),
            )
            for op in tuple(getattr(artifact, "ops", ()) or ())
        )
        artifact = replace(artifact, ops=ops)
    return replace(trace_plan, root_module=model, runtime_artifact=artifact)


def _move_template_tensors_to_device(value: Any, device: Any) -> Any:
    if isinstance(value, ConstantTensorArg):
        return replace(value, value=value.value.to(device))
    if isinstance(value, torch.device):
        return torch.device(device)
    if hasattr(value, "device") and hasattr(value, "to"):
        try:
            return value.to(device)
        except Exception:
            return value
    if isinstance(value, tuple):
        return tuple(_move_template_tensors_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_template_tensors_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {
            key: _move_template_tensors_to_device(item, device)
            for key, item in value.items()
        }
    if isinstance(value, slice):
        return slice(
            _move_template_tensors_to_device(value.start, device),
            _move_template_tensors_to_device(value.stop, device),
            _move_template_tensors_to_device(value.step, device),
        )
    return value


def _rebind_candidate_device(candidate: Any, model: Any) -> Any:
    if candidate is None:
        return None
    target_device = _model_device(model)
    if target_device is None:
        return candidate

    def _rebind_spec(spec: Any) -> Any:
        try:
            return replace(spec, device_type=str(target_device.type))
        except Exception:
            return spec

    boundary_schema = getattr(candidate, "boundary_schema", None)
    if isinstance(boundary_schema, dict):
        boundary_schema = {
            str(label): _rebind_spec(spec)
            for label, spec in boundary_schema.items()
        }

    boundary_value_schema = getattr(candidate, "boundary_value_schema", None)
    if isinstance(boundary_value_schema, dict):
        boundary_value_schema = {
            str(label): replace(spec, tensor_spec=_rebind_spec(getattr(spec, "tensor_spec", None)))
            for label, spec in boundary_value_schema.items()
        }

    try:
        return replace(
            candidate,
            boundary_schema=boundary_schema if boundary_schema is not None else getattr(candidate, "boundary_schema", None),
            boundary_value_schema=boundary_value_schema if boundary_value_schema is not None else getattr(candidate, "boundary_value_schema", None),
        )
    except TypeError:
        return candidate


_PROCESS_FIXED_SPLIT_RUNTIME_TEMPLATE_CACHE = FixedSplitRuntimeTemplateCache()


def get_fixed_split_runtime_template_cache() -> FixedSplitRuntimeTemplateCache:
    return _PROCESS_FIXED_SPLIT_RUNTIME_TEMPLATE_CACHE


__all__ = [
    "FIXED_SPLIT_RUNTIME_TEMPLATE_CACHE_VERSION",
    "FixedSplitRuntimeTemplate",
    "FixedSplitRuntimeTemplateCache",
    "FixedSplitRuntimeTemplateKey",
    "FixedSplitRuntimeTemplateLookup",
    "bind_request_runtime_from_template",
    "fixed_split_runtime_template_key",
    "get_fixed_split_runtime_template_cache",
]
