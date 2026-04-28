from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from ariadne import BoundaryPayload, SplitRuntime, SplitSpec

from model_management.payload import deserialize_boundary_payload, serialize_boundary_payload
from model_management.split_candidate import CandidateProfile, SplitCandidate
from model_management.split_runtime import (
    compare_outputs,
    make_split_spec,
    prepare_split_runtime,
    run_batch_prefix,
    run_batch_suffix,
    train_batch_suffix,
)


def _first_tensor_batch_size(value: Any) -> int | None:
    if isinstance(value, torch.Tensor) and value.ndim > 0:
        return int(value.shape[0])
    if isinstance(value, Mapping):
        for item in value.values():
            found = _first_tensor_batch_size(item)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor_batch_size(item)
            if found is not None:
                return found
    return None


@dataclass
class LayerInfo:
    index: int
    name: str
    module_type: str
    output_shape: tuple[int, ...] | None = None
    trainable_params: int = 0


@dataclass
class LayerProfile:
    layer: LayerInfo
    estimated_flops: float = 0.0
    estimated_bytes: int = 0


@dataclass
class SplitPointSelector:
    candidates: list[SplitCandidate] = field(default_factory=list)

    def best(self) -> SplitCandidate | None:
        return self.candidates[0] if self.candidates else None


@dataclass
class SplitCandidateSelector(SplitPointSelector):
    pass


def _candidate_from_runtime(runtime: SplitRuntime, split_spec: SplitSpec) -> SplitCandidate:
    split_id = str(getattr(runtime, "split_id", split_spec.boundary))
    boundary = split_id.split("after:", 1)[-1]
    return SplitCandidate(
        candidate_id=split_id,
        edge_nodes=[boundary],
        cloud_nodes=[],
        boundary_edges=[],
        boundary_tensor_labels=[boundary],
        edge_input_labels=[],
        cloud_input_labels=[],
        cloud_output_labels=[],
        estimated_edge_flops=0.0,
        estimated_cloud_flops=0.0,
        estimated_payload_bytes=0,
        estimated_privacy_risk=0.0,
        estimated_latency=0.0,
        is_trainable_tail=True,
        is_validated=True,
        legacy_layer_index=None,
        boundary_count=1,
        metadata={
            "runtime": "ariadne",
            "graph_signature": getattr(runtime, "graph_signature", None),
            "split_spec": {
                "boundary": split_spec.boundary,
                "dynamic_batch": split_spec.dynamic_batch,
                "trace_batch_mode": split_spec.trace_batch_mode,
            },
        },
    )


def build_candidate_descriptor(candidate: SplitCandidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "boundary_tensor_labels": list(candidate.boundary_tensor_labels),
        "legacy_layer_index": candidate.legacy_layer_index,
        "metadata": dict(candidate.metadata),
    }


def reconstruct_candidate_from_descriptor(
    graph: Any,
    descriptor: Mapping[str, Any] | None,
    *,
    source: str | None = None,
) -> SplitCandidate | None:
    del graph
    if not descriptor:
        return None
    labels = list(descriptor.get("boundary_tensor_labels", []))
    candidate_id = str(descriptor.get("candidate_id") or (labels[-1] if labels else "after:auto"))
    return SplitCandidate(
        candidate_id=candidate_id,
        edge_nodes=labels,
        cloud_nodes=[],
        boundary_edges=[],
        boundary_tensor_labels=labels,
        edge_input_labels=[],
        cloud_input_labels=[],
        cloud_output_labels=[],
        estimated_edge_flops=0.0,
        estimated_cloud_flops=0.0,
        estimated_payload_bytes=0,
        estimated_privacy_risk=0.0,
        estimated_latency=0.0,
        is_trainable_tail=True,
        is_validated=True,
        legacy_layer_index=descriptor.get("legacy_layer_index"),
        boundary_count=len(labels),
        metadata={**dict(descriptor.get("metadata", {})), "source": source or "descriptor"},
    )


class UniversalModelSplitter:
    """Thin Plank-road facade over Ariadne SplitRuntime.

    This class intentionally does not own a graph replay engine. It keeps the
    older orchestration entry points where the surrounding code still calls
    them, but every runtime operation delegates to Ariadne.
    """

    def __init__(self, *, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.model: torch.nn.Module | None = None
        self.runtime: SplitRuntime | None = None
        self.split_spec: SplitSpec | None = None
        self.current_candidate: SplitCandidate | None = None
        self.candidates: list[SplitCandidate] = []
        self.graph: str | None = None
        self.history = None
        self.trace_timings: dict[str, float] = {}
        self.trace_used_output_fallback = False
        self.trainability_loss_fn = None
        self.model_name: str | None = None
        self.model_family: str | None = None

    def trace(
        self,
        model: torch.nn.Module,
        sample_input: Any,
        sample_kwargs: Mapping[str, Any] | None = None,
        *,
        split_spec: SplitSpec | None = None,
        boundary: str = "auto",
        mode: str = "generated_eager",
        model_name: str | None = None,
        model_family: str | None = None,
        **_: Any,
    ) -> "UniversalModelSplitter":
        if sample_kwargs:
            raise RuntimeError("Ariadne split runtime currently expects positional example inputs.")
        self.model = model
        self.model_name = model_name
        self.model_family = model_family
        trace_batch_size = _first_tensor_batch_size(sample_input) or 1
        trace_batch_mode = "batch_gt1" if trace_batch_size > 1 else "batch_1"
        dynamic_batch = (2, 64) if trace_batch_size > 1 else (1, 64)
        self.split_spec = split_spec or make_split_spec(
            boundary,
            dynamic_batch=dynamic_batch,
            trainable=True,
            trace_batch_mode=trace_batch_mode,
            model_family=model_family,
        )
        self.runtime = prepare_split_runtime(
            model,
            sample_input,
            self.split_spec,
            mode=mode,
        )
        self.graph = str(getattr(self.runtime, "graph_signature", ""))
        self.current_candidate = _candidate_from_runtime(self.runtime, self.split_spec)
        self.candidates = [self.current_candidate]
        return self

    def trace_graph(
        self,
        model: torch.nn.Module,
        sample_input: Any,
        sample_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> "UniversalModelSplitter":
        return self.trace(model, sample_input, sample_kwargs=sample_kwargs, **kwargs)

    def bind_runtime(
        self,
        runtime: SplitRuntime,
        *,
        model: torch.nn.Module | None = None,
        split_spec: SplitSpec | None = None,
    ) -> "UniversalModelSplitter":
        self.runtime = runtime
        self.model = model
        self.split_spec = split_spec or SplitSpec(boundary=getattr(runtime, "split_id", "auto"))
        self.graph = str(getattr(runtime, "graph_signature", ""))
        self.current_candidate = _candidate_from_runtime(runtime, self.split_spec)
        self.candidates = [self.current_candidate]
        return self

    def bind_graph(self, model: torch.nn.Module, graph: Any, **_: Any) -> "UniversalModelSplitter":
        if isinstance(graph, SplitRuntime):
            return self.bind_runtime(graph, model=model)
        raise RuntimeError("Graph templates are no longer supported; bind an Ariadne SplitRuntime.")

    def _ensure_runtime(self) -> SplitRuntime:
        if self.runtime is None:
            raise RuntimeError(
                "prepare_split_runtime() or trace() must be called before split execution."
            )
        return self.runtime

    def split(
        self,
        *,
        candidate: SplitCandidate | None = None,
        candidate_id: str | None = None,
        layer_index: int | None = None,
        layer_label: str | None = None,
        boundary_tensor_labels: list[str] | None = None,
    ) -> SplitCandidate:
        del layer_index, layer_label, boundary_tensor_labels
        if candidate is not None:
            self.current_candidate = candidate
            return candidate
        chosen = self.current_candidate
        if chosen is None:
            runtime = self._ensure_runtime()
            chosen = _candidate_from_runtime(runtime, self.split_spec or SplitSpec(boundary="auto"))
            self.current_candidate = chosen
        if candidate_id is not None and chosen.candidate_id != candidate_id:
            raise KeyError(
                "Ariadne runtime exposes "
                f"split_id={chosen.candidate_id!r}, not {candidate_id!r}."
            )
        return chosen

    def enumerate_candidates(self, **_: Any) -> list[SplitCandidate]:
        return list(self.candidates or [self.split()])

    def bind_candidate_descriptor(
        self,
        descriptor: Mapping[str, Any],
        *,
        include_in_candidates: bool = False,
    ) -> SplitCandidate:
        candidate = reconstruct_candidate_from_descriptor(None, descriptor)
        if candidate is None:
            raise RuntimeError("Could not bind empty split candidate descriptor.")
        self.current_candidate = candidate
        if include_in_candidates:
            self.candidates = [candidate]
        return candidate

    def edge_forward(
        self,
        *args: Any,
        candidate: SplitCandidate | None = None,
        **kwargs: Any,
    ) -> BoundaryPayload:
        del candidate
        if kwargs:
            raise RuntimeError("Ariadne prefix execution expects positional runtime inputs.")
        return run_batch_prefix(
            self._ensure_runtime(),
            *args,
            model_name=self.model_name,
            model_family=self.model_family,
        )

    run_prefix = edge_forward

    def cloud_forward(
        self,
        payload: BoundaryPayload,
        *args: Any,
        candidate: SplitCandidate | None = None,
        **kwargs: Any,
    ) -> Any:
        del args, candidate, kwargs
        return run_batch_suffix(
            self._ensure_runtime(),
            payload,
            model_name=self.model_name,
            model_family=self.model_family,
        )

    run_suffix = cloud_forward

    def cloud_train_step(
        self,
        payload: BoundaryPayload,
        targets: Any = None,
        *,
        loss_fn=None,
        optimizer: torch.optim.Optimizer | None = None,
        candidate: SplitCandidate | None = None,
        **_: Any,
    ) -> tuple[None, torch.Tensor]:
        del candidate
        loss, _grads = train_batch_suffix(
            self._ensure_runtime(),
            payload,
            targets,
            loss_fn=loss_fn or self.trainability_loss_fn,
            optimizer=optimizer,
            model_name=self.model_name,
            model_family=self.model_family,
        )
        return None, loss

    def train_suffix(
        self,
        boundary: BoundaryPayload,
        targets: Any,
        *,
        loss_fn=None,
        optimizer=None,
    ):
        return train_batch_suffix(
            self._ensure_runtime(),
            boundary,
            targets,
            loss_fn=loss_fn or self.trainability_loss_fn,
            optimizer=optimizer,
            model_name=self.model_name,
            model_family=self.model_family,
        )

    def replay_inference(self, sample_input: Any, *, return_split_output: bool = False):
        payload = self.edge_forward(sample_input)
        outputs = self.cloud_forward(payload)
        return (outputs, payload) if return_split_output else outputs

    def full_forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.model is None:
            raise RuntimeError("No model is bound.")
        return self.model(*args, **kwargs)

    full_replay = full_forward

    def validate_candidate(
        self,
        candidate: SplitCandidate | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        chosen = candidate or self.current_candidate
        return {
            "success": self.runtime is not None,
            "candidate_id": getattr(chosen, "candidate_id", None),
            "runtime": "ariadne",
        }

    def freeze_head(self, chosen: SplitCandidate | None = None) -> None:
        del chosen

    def unfreeze_tail(self, chosen: SplitCandidate | None = None) -> None:
        del chosen
        if self.model is not None:
            for parameter in self.model.parameters():
                parameter.requires_grad_(True)

    def get_tail_trainable_params(
        self,
        chosen: SplitCandidate | None = None,
    ) -> Iterable[torch.nn.Parameter]:
        del chosen
        if self.model is None:
            return []
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]


def extract_split_features(splitter: UniversalModelSplitter, sample_input: Any) -> BoundaryPayload:
    return splitter.edge_forward(sample_input)


def _feature_path(cache_path: str, frame_index: Any) -> str:
    return os.path.join(cache_path, "features", f"{frame_index}.pt")


def save_split_feature_cache(
    cache_path: str,
    frame_index: Any,
    intermediate: BoundaryPayload,
    **record_fields: Any,
) -> None:
    os.makedirs(os.path.join(cache_path, "features"), exist_ok=True)
    extra_metadata = dict(record_fields.pop("extra_metadata", {}) or {})
    record = {
        "intermediate": intermediate,
        "candidate_id": getattr(intermediate, "candidate_id", None)
        or getattr(intermediate, "split_id", None),
        "boundary_tensor_labels": getattr(
            intermediate,
            "boundary_tensor_labels",
            list(getattr(intermediate, "tensors", {}).keys()),
        ),
        "split_index": getattr(intermediate, "split_index", None),
        "split_label": getattr(intermediate, "split_label", None)
        or getattr(intermediate, "split_id", None),
        **record_fields,
        **extra_metadata,
    }
    torch.save(record, _feature_path(cache_path, frame_index))


def load_split_feature_cache(cache_path: str, frame_index: Any) -> dict[str, Any]:
    path = _feature_path(cache_path, frame_index)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    record = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(record, dict):
        raise TypeError(f"Unsupported split feature cache record: {type(record)!r}")
    return record


def universal_split_retrain(
    *,
    model: torch.nn.Module,
    sample_input: Any,
    cache_path: str,
    all_indices: list[Any],
    gt_annotations: Mapping[Any, Any] | None = None,
    device: str | torch.device = "cpu",
    num_epoch: int = 1,
    learning_rate: float = 1e-4,
    loss_fn=None,
    splitter: UniversalModelSplitter | None = None,
    batch_size: int = 2,
    **_: Any,
) -> list[float]:
    if loss_fn is None:
        raise RuntimeError("Split-tail training requires an explicit loss function.")
    runtime = splitter or UniversalModelSplitter(device=device).trace(model, sample_input)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(params, lr=float(learning_rate)) if params else None
    losses: list[float] = []
    annotations = dict(gt_annotations or {})

    for _epoch in range(int(num_epoch)):
        epoch_losses: list[float] = []
        for start in range(0, len(all_indices), int(batch_size)):
            batch_indices = list(all_indices[start : start + int(batch_size)])
            if not batch_indices:
                continue
            records = [load_split_feature_cache(cache_path, index) for index in batch_indices]
            boundaries = [record.get("intermediate") for record in records]
            if not boundaries or not all(
                isinstance(boundary, BoundaryPayload) for boundary in boundaries
            ):
                raise RuntimeError(
                    "Split-tail training requires cached batch BoundaryPayload records; "
                    "per-sample payload batching is intentionally unsupported."
                )
            boundary = boundaries[0]
            if int(boundary.batch_size) != len(batch_indices):
                raise RuntimeError(
                    "Cached BoundaryPayload batch size does not match training batch "
                    f"(payload_batch={boundary.batch_size}, requested_batch={len(batch_indices)})."
                )
            targets = [annotations.get(index) for index in batch_indices]
            loss, _grads = runtime.train_suffix(
                boundary,
                targets,
                loss_fn=loss_fn,
                optimizer=optimizer,
            )
            epoch_losses.append(float(loss.detach().cpu().item()))
        if not epoch_losses:
            raise RuntimeError("Split retraining did not produce any finite batch loss.")
        losses.append(sum(epoch_losses) / len(epoch_losses))
    return losses


__all__ = [
    "BoundaryPayload",
    "CandidateProfile",
    "LayerInfo",
    "LayerProfile",
    "SplitCandidate",
    "SplitCandidateSelector",
    "SplitPointSelector",
    "SplitRuntime",
    "SplitSpec",
    "UniversalModelSplitter",
    "build_candidate_descriptor",
    "compare_outputs",
    "deserialize_boundary_payload",
    "extract_split_features",
    "load_split_feature_cache",
    "reconstruct_candidate_from_descriptor",
    "save_split_feature_cache",
    "serialize_boundary_payload",
    "universal_split_retrain",
]
