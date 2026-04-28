from __future__ import annotations

import threading
import time
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any

from loguru import logger

from .ariadne_runtime import SplitRuntime, SplitSpec
from .runtime_cache import RuntimeCacheKey, make_runtime_cache_key

FIXED_SPLIT_RUNTIME_TEMPLATE_CACHE_VERSION = 2


@dataclass(frozen=True)
class FixedSplitRuntimeTemplateKey(RuntimeCacheKey):
    version: int = FIXED_SPLIT_RUNTIME_TEMPLATE_CACHE_VERSION

    def as_dict(self) -> dict[str, object]:
        payload = super().as_dict()
        payload["version"] = int(self.version)
        return payload


def fixed_split_runtime_template_key(
    *,
    model_name: str,
    model_family: str,
    split_spec: SplitSpec,
    example_inputs: Any,
    graph_signature: str | None = None,
    split_plan_hash: str | None = None,
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
    return FixedSplitRuntimeTemplateKey(**key.__dict__)


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
        The split runtime from the template.
    """
    _ = model, device  # Reserved parameters for future extensibility
    return template.runtime


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
