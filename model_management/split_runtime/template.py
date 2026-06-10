from __future__ import annotations

import threading
import time
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from typing import Any

from loguru import logger

from .runtime_cache import RuntimeCacheKey, make_runtime_cache_key
from .torchlens_native_runtime import SplitRuntime, SplitSpec, prepare_split_runtime

FIXED_SPLIT_RUNTIME_TEMPLATE_CACHE_VERSION = 6


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
            payload["runtime_batch_validation_signature"] = self.runtime_batch_validation_signature
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
        trace_batch_size=(None if trace_batch_size is None else max(1, int(trace_batch_size))),
        validated_batch_max=(
            None if validated_batch_max is None else max(1, int(validated_batch_max))
        ),
        runtime_batch_validation_signature=(
            str(runtime_batch_validation_signature) if runtime_batch_validation_signature else None
        ),
    )


@dataclass(frozen=True)
class FixedSplitRuntimeTemplate:
    cache_key: FixedSplitRuntimeTemplateKey
    # TorchLens native executable runtimes are not rebindable across model
    # instances. This runtime is kept only as a read-only trace/layout artifact.
    runtime: SplitRuntime
    split_spec: SplitSpec
    model_name: str
    model_family: str
    graph_signature: str | None
    symbolic_input_schema_hash: str
    split_plan_hash: str
    mode: str = "generated_eager"
    runtime_device: str = ""
    candidate_descriptor: Mapping[str, object] | None = None
    runtime_contract: Mapping[str, object] | None = None
    boundary_tensor_labels: tuple[str, ...] = ()
    boundary_schema: Mapping[str, object] | None = None


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
                    "[FixedSplitCL] TorchLens runtime template cache hit (key={}).",
                    cache_key,
                )
                return FixedSplitRuntimeTemplateLookup(template=cached, cache_status="hit")

            inflight = self._inflight.get(cache_key)
            if inflight is None:
                inflight = _InflightTemplateBuild(event=threading.Event())
                self._inflight[cache_key] = inflight
                build_owner = True
                logger.info(
                    "[FixedSplitCL] TorchLens runtime template cache miss (key={}).",
                    cache_key,
                )
            else:
                build_owner = False
                logger.info(
                    "[FixedSplitCL] Waiting for TorchLens runtime template build (key={}).",
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
                    "TorchLens runtime template build completed without a template "
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
            "[FixedSplitCL] TorchLens runtime template cold build completed in {:.3f}s (key={}).",
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
    example_inputs: Any | None = None,
    device: str | None = None,
) -> SplitRuntime:
    del device
    if model is None or model is getattr(template.runtime, "model", None):
        return template.runtime
    if example_inputs is None:
        raise RuntimeError(
            "TorchLens runtime cannot be rebound by deepcopy. Provide example_inputs "
            "so bind_request_runtime_from_template can prepare a request-local "
            "SplitRuntime for the supplied model."
        )
    return prepare_split_runtime(
        model,
        example_inputs,
        template.split_spec,
        mode=template.mode,
    )


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
