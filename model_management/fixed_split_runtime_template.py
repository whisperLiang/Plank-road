from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Hashable, Mapping

import torch
from loguru import logger

from model_management.graph_ir import GraphIR
from model_management.split_candidate import SplitCandidate
from model_management.universal_model_split import (
    UniversalModelSplitter,
    build_candidate_descriptor,
    reconstruct_candidate_from_descriptor,
)


FIXED_SPLIT_RUNTIME_TEMPLATE_CACHE_VERSION = 1


def _stable_json_dumps(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _key_log_string(key: Hashable) -> str:
    if hasattr(key, "log_string") and callable(getattr(key, "log_string")):
        return str(key.log_string())
    if hasattr(key, "to_log_payload") and callable(getattr(key, "to_log_payload")):
        return _stable_json_dumps(key.to_log_payload())
    if isinstance(key, (dict, list, tuple)):
        return _stable_json_dumps(key)
    return str(key)


@dataclass(frozen=True)
class FixedSplitRuntimeTemplateKey:
    """Strict cache key for process-local fixed-split runtime templates.

    Weight fingerprints are intentionally excluded. Retraining mutates weights,
    but the traced graph topology and the uploaded fixed split plan remain
    reusable as long as the model family and trace-shape contract stay stable.
    """

    model_name: str
    trace_image_size: tuple[int, int] | None
    trace_input_shape: tuple[int, ...] | None
    trace_batch_size: int
    split_plan_hash: str
    version: int = FIXED_SPLIT_RUNTIME_TEMPLATE_CACHE_VERSION

    def as_dict(self) -> dict[str, object]:
        return {
            "version": int(self.version),
            "model_name": str(self.model_name),
            "trace_image_size": (
                list(self.trace_image_size)
                if self.trace_image_size is not None
                else None
            ),
            "trace_input_shape": (
                list(self.trace_input_shape)
                if self.trace_input_shape is not None
                else None
            ),
            "trace_batch_size": int(self.trace_batch_size),
            "split_plan_hash": str(self.split_plan_hash),
        }

    def log_string(self) -> str:
        return _stable_json_dumps(self.as_dict())

    def to_log_payload(self) -> dict[str, object]:
        return self.as_dict()


@dataclass(frozen=True)
class FixedSplitRuntimeTemplate:
    cache_key: FixedSplitRuntimeTemplateKey | None
    graph: GraphIR
    candidate_descriptor: Mapping[str, Any]
    trace_signature: str
    trace_timings: Mapping[str, float]
    trace_used_output_fallback: bool
    trace_image_size: tuple[int, int] | None
    trace_input_shape: tuple[int, ...] | None
    trace_batch_size: int
    split_plan_hash: str
    candidate_recovery_mode: str | None = None


@dataclass(frozen=True)
class FixedSplitRuntimeTemplateLookup:
    template: FixedSplitRuntimeTemplate
    cache_status: str
    wait_time_sec: float = 0.0
    cold_build_time_sec: float = 0.0


@dataclass
class _InflightTemplateBuild:
    event: threading.Event
    waiters: int = 0
    template: FixedSplitRuntimeTemplate | None = None
    error: BaseException | None = None


def describe_split_candidate(candidate: SplitCandidate) -> dict[str, Any]:
    return build_candidate_descriptor(candidate)


def restore_split_candidate(
    graph: GraphIR,
    descriptor: Mapping[str, Any] | None,
) -> SplitCandidate:
    candidate = reconstruct_candidate_from_descriptor(
        graph,
        descriptor,
        source="fixed_split_runtime_template",
    )
    if candidate is None:
        raise RuntimeError(
            "Could not restore fixed split candidate from the cached runtime template."
        )
    boundary_tensor_labels = list(dict(descriptor or {}).get("boundary_tensor_labels", []))
    if boundary_tensor_labels and list(candidate.boundary_tensor_labels) != boundary_tensor_labels:
        raise RuntimeError(
            "Restored fixed split candidate does not match cached boundary labels. "
            f"expected={boundary_tensor_labels!r}, actual={list(candidate.boundary_tensor_labels)!r}"
        )
    return candidate


def compute_graph_trace_signature(graph: GraphIR) -> str:
    digest: list[dict[str, object]] = []
    for label in getattr(graph, "relevant_labels", ()) or ():
        node = graph.nodes[label]
        digest.append(
            {
                "label": str(label),
                "shape": list(getattr(node, "tensor_shape", ()) or ()) or None,
                "module": getattr(node, "containing_module", None),
                "trainable": bool(getattr(node, "has_trainable_params", False)),
            }
        )
    raw = json.dumps(digest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def freeze_trace_timings(trace_timings: Mapping[str, float] | None) -> Mapping[str, float]:
    payload = {
        str(stage): float(elapsed)
        for stage, elapsed in dict(trace_timings or {}).items()
    }
    return MappingProxyType(payload)


def bind_request_splitter_from_template(
    runtime_model: torch.nn.Module,
    template: FixedSplitRuntimeTemplate,
    *,
    device: str | torch.device = "cpu",
) -> tuple[UniversalModelSplitter, SplitCandidate]:
    # Each request gets a fresh splitter and candidate object. Only the graph
    # template is shared, and it is treated as read-only across requests.
    splitter = UniversalModelSplitter(device=device)
    splitter.bind_graph(
        runtime_model,
        template.graph,
        candidates=[],
        current_candidate=None,
        trace_timings=dict(template.trace_timings),
        trace_used_output_fallback=template.trace_used_output_fallback,
        candidate_enumeration_config=None,
    )
    candidate = splitter.bind_candidate_descriptor(
        dict(template.candidate_descriptor),
        include_in_candidates=True,
    )
    return splitter, candidate


def bind_request_runtime_from_template(
    template: FixedSplitRuntimeTemplate,
    *,
    model: torch.nn.Module,
    device: str | torch.device = "cpu",
) -> tuple[UniversalModelSplitter, SplitCandidate]:
    return bind_request_splitter_from_template(model, template, device=device)


class FixedSplitRuntimeTemplateCache:
    """Process-local immutable template cache with per-key single-flight.

    The cache stores only read-only graph and candidate template data. Mutable
    execution state stays on request-local splitters created from the template.
    """

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
            template = self._templates.get(cache_key)
            if template is not None:
                logger.info(
                    "[FixedSplitCL] runtime template cache hit (key={}).",
                    _key_log_string(cache_key),
                )
                return FixedSplitRuntimeTemplateLookup(
                    template=template,
                    cache_status="hit",
                )

            inflight = self._inflight.get(cache_key)
            if inflight is None:
                inflight = _InflightTemplateBuild(event=threading.Event())
                self._inflight[cache_key] = inflight
                build_owner = True
                logger.info(
                    "[FixedSplitCL] runtime template cache miss; starting cold build (key={}).",
                    _key_log_string(cache_key),
                )
            else:
                inflight.waiters += 1
                build_owner = False
                logger.info(
                    "[FixedSplitCL] runtime template cache wait-for-existing-build (key={}, waiters={}).",
                    _key_log_string(cache_key),
                    inflight.waiters,
                )

        if not build_owner:
            wait_started = time.perf_counter()
            inflight.event.wait()
            wait_elapsed = time.perf_counter() - wait_started
            if inflight.error is not None:
                raise inflight.error
            if inflight.template is None:
                raise RuntimeError(
                    "Fixed split runtime template build completed without a template."
                )
            logger.info(
                "[FixedSplitCL] runtime template wait complete in {:.3f}s (key={}).",
                wait_elapsed,
                _key_log_string(cache_key),
            )
            return FixedSplitRuntimeTemplateLookup(
                template=inflight.template,
                cache_status="wait",
                wait_time_sec=wait_elapsed,
            )

        build_started = time.perf_counter()
        try:
            template = builder()
        except BaseException as exc:
            with self._lock:
                inflight.error = exc
                self._inflight.pop(cache_key, None)
                inflight.event.set()
            raise

        cold_build_time = time.perf_counter() - build_started
        with self._lock:
            self._templates[cache_key] = template
            inflight.template = template
            self._inflight.pop(cache_key, None)
            inflight.event.set()
        logger.info(
            "[FixedSplitCL] runtime template cold build completed in {:.3f}s (key={}).",
            cold_build_time,
            _key_log_string(cache_key),
        )
        return FixedSplitRuntimeTemplateLookup(
            template=template,
            cache_status="miss",
            cold_build_time_sec=cold_build_time,
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


_PROCESS_FIXED_SPLIT_RUNTIME_TEMPLATE_CACHE = FixedSplitRuntimeTemplateCache()


def get_fixed_split_runtime_template_cache() -> FixedSplitRuntimeTemplateCache:
    return _PROCESS_FIXED_SPLIT_RUNTIME_TEMPLATE_CACHE


__all__ = [
    "FIXED_SPLIT_RUNTIME_TEMPLATE_CACHE_VERSION",
    "FixedSplitRuntimeTemplateKey",
    "FixedSplitRuntimeTemplate",
    "FixedSplitRuntimeTemplateLookup",
    "FixedSplitRuntimeTemplateCache",
    "bind_request_runtime_from_template",
    "bind_request_splitter_from_template",
    "compute_graph_trace_signature",
    "describe_split_candidate",
    "freeze_trace_timings",
    "get_fixed_split_runtime_template_cache",
]
