"""
Graph-based split runtime for arbitrary TorchLens-traceable PyTorch models.

The public facade remains `UniversalModelSplitter`, but the implementation is
candidate-driven:
  - a split is a validated graph partition candidate
  - replay is dependency-driven over a graph IR
  - payloads are minimal boundary tensor maps
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from model_management.candidate_generator import build_candidate_from_edge_seed, enumerate_candidates
from model_management.candidate_profiler import profile_candidates
from model_management.candidate_selector import SplitCandidateSelector, SplitPointSelector
from model_management.payload import SplitPayload
from model_management.split_candidate import CandidateProfile, SplitCandidate
from model_management.split_runtime import GraphSplitRuntime

try:
    import torchlens as tl

    _HAS_TORCHLENS = True
except ImportError:  # pragma: no cover - import guard
    _HAS_TORCHLENS = False


@dataclass
class LayerInfo:
    index: int
    label: str
    layer_type: str
    func_name: str
    output_shape: tuple[int, ...] | None
    has_params: bool
    num_params: int
    module_name: str | None
    is_input: bool = False
    is_output: bool = False


@dataclass
class LayerProfile:
    index: int
    label: str
    cumulative_flops: float
    smashed_data_size: int
    output_shape: tuple[int, ...] | None
    privacy_leakage: float


def _move_nested(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, list):
        return [_move_nested(item, device) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_move_nested(item, device) for item in obj)
    if isinstance(obj, dict):
        return {key: _move_nested(value, device) for key, value in obj.items()}
    return obj


def _cache_feature_dir(cache_path: str) -> str:
    feature_dir = os.path.join(cache_path, "features")
    os.makedirs(feature_dir, exist_ok=True)
    return feature_dir


class UniversalModelSplitter(GraphSplitRuntime):
    def __init__(self, *, device: str | torch.device = "cpu") -> None:
        super().__init__(device=device)
        self.selected_candidate_id: str | None = None

    def trace(
        self,
        model: torch.nn.Module,
        sample_input: Any,
        sample_kwargs: Mapping[str, Any] | None = None,
    ) -> "UniversalModelSplitter":
        super().trace(model, sample_input, sample_kwargs=sample_kwargs)
        if self.current_candidate is not None:
            self.selected_candidate_id = self.current_candidate.candidate_id
        return self

    def list_layers(self) -> list[LayerInfo]:
        _, graph = self._ensure_ready()
        infos: list[LayerInfo] = []
        for index, label in enumerate(graph.topological_order):
            node = graph.nodes[label]
            infos.append(
                LayerInfo(
                    index=index,
                    label=label,
                    layer_type=node.node_type,
                    func_name=node.func_name,
                    output_shape=node.tensor_shape,
                    has_params=bool(node.parameter_refs),
                    num_params=len(node.parameter_refs),
                    module_name=node.containing_module,
                    is_input=node.is_input,
                    is_output=node.is_output,
                )
            )
        return infos

    def num_layers(self) -> int:
        _, graph = self._ensure_ready()
        return len(graph.topological_order)

    @property
    def split_index(self) -> int | None:
        candidate = self.current_candidate
        return None if candidate is None else candidate.legacy_layer_index

    @property
    def split_candidate_id(self) -> str | None:
        candidate = self.current_candidate
        return None if candidate is None else candidate.candidate_id

    def enumerate_candidates(
        self,
        *,
        max_candidates: int = 24,
        max_boundary_count: int = 8,
        max_payload_bytes: int = 32 * 1024 * 1024,
    ) -> list[SplitCandidate]:
        _, graph = self._ensure_ready()
        self.candidates = enumerate_candidates(
            graph,
            max_candidates=max_candidates,
            max_boundary_count=max_boundary_count,
            max_payload_bytes=max_payload_bytes,
        )
        return list(self.candidates)

    def _candidate_from_layer_index(self, layer_index: int) -> SplitCandidate:
        _, graph = self._ensure_ready()
        relevant = graph.relevant_labels
        if not relevant:
            raise RuntimeError("No output-relevant graph nodes are available.")
        layer_index = max(0, min(layer_index, len(relevant) - 2))
        seed = relevant[: layer_index + 1]
        candidate = build_candidate_from_edge_seed(
            graph,
            candidate_id=f"legacy_{layer_index}",
            edge_seed_nodes=seed,
            legacy_layer_index=layer_index,
            metadata={"source": "legacy_layer_index"},
        )
        if candidate is None:
            raise RuntimeError(f"Could not build a valid graph partition from layer_index={layer_index}.")
        return candidate

    def split(
        self,
        *,
        candidate_id: str | None = None,
        layer_index: int | None = None,
        layer_label: str | None = None,
        candidate: SplitCandidate | None = None,
    ) -> SplitCandidate:
        if candidate is not None:
            self.current_candidate = candidate
            self.selected_candidate_id = candidate.candidate_id
            return candidate

        if candidate_id is not None:
            self.current_candidate = self._candidate_or_default(candidate_id)
        elif layer_label is not None:
            matches = [
                item
                for item in self.candidates
                if layer_label in item.boundary_tensor_labels or layer_label in item.edge_nodes
            ]
            if not matches:
                raise KeyError(f"No candidate references layer label {layer_label!r}.")
            self.current_candidate = matches[0]
        elif layer_index is not None:
            self.current_candidate = self._candidate_from_layer_index(layer_index)
        elif self.current_candidate is None and self.candidates:
            self.current_candidate = self.candidates[0]

        if self.current_candidate is None:
            raise RuntimeError("No split candidate is available.")
        self.selected_candidate_id = self.current_candidate.candidate_id
        return self.current_candidate

    def edge_forward(self, *args: Any, **kwargs: Any) -> SplitPayload | torch.Tensor:
        payload = super().edge_forward(*args, **kwargs)
        return payload

    def cloud_forward(self, payload: Any, *args: Any, **kwargs: Any) -> Any:
        return super().cloud_forward(payload, *args, **kwargs)

    def full_forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.full_replay(*args, **kwargs)

    def profile_layers(self) -> list[LayerProfile]:
        _, graph = self._ensure_ready()
        profiles: list[LayerProfile] = []
        cumulative_flops = 0.0
        for index, label in enumerate(graph.relevant_labels):
            node = graph.nodes[label]
            cumulative_flops += node.estimated_flops
            privacy = (node.estimated_bytes / 1024.0) / max(1, node.depth_from_input + 1)
            profiles.append(
                LayerProfile(
                    index=index,
                    label=label,
                    cumulative_flops=cumulative_flops,
                    smashed_data_size=node.estimated_bytes,
                    output_shape=node.tensor_shape,
                    privacy_leakage=privacy,
                )
            )
        return profiles

    def profile_candidates(
        self,
        *,
        validate: bool = True,
        validation_runs: int = 1,
    ) -> list[CandidateProfile]:
        return profile_candidates(self, self.candidates, validate=validate, validation_runs=validation_runs)

    def create_split_selector(
        self,
        profiles: Sequence[CandidateProfile] | None = None,
        *,
        alpha: float = 0.35,
        epsilon: float = 0.05,
    ) -> SplitCandidateSelector:
        return SplitCandidateSelector(self.candidates, profiles or [], alpha=alpha, epsilon=epsilon)

    @staticmethod
    def serialise_intermediate(payload: SplitPayload | torch.Tensor | Mapping[str, torch.Tensor], *, compress: bool = False) -> bytes:
        if isinstance(payload, SplitPayload):
            return payload.cpu().serialize(compress=compress)
        if isinstance(payload, torch.Tensor):
            payload = SplitPayload.from_mapping({"payload": payload.cpu()}, primary_label="payload")
            return payload.serialize(compress=compress)
        if isinstance(payload, Mapping):
            payload = SplitPayload.from_mapping(
                OrderedDict((str(label), tensor.cpu()) for label, tensor in payload.items()),
            )
            return payload.serialize(compress=compress)
        raise TypeError(f"Unsupported payload type: {type(payload)!r}")

    @staticmethod
    def deserialise_intermediate(data: bytes, *, compressed: bool = False) -> SplitPayload:
        return SplitPayload.deserialize(data, compressed=compressed)

    def split_retrain(
        self,
        train_data: Sequence[tuple[Any, Any]],
        loss_fn=None,
        *,
        num_epochs: int = 1,
        lr: float = 1e-3,
        candidate: SplitCandidate | str | None = None,
    ) -> list[float]:
        chosen = self._candidate_or_default(candidate)
        self.freeze_head(chosen)
        self.unfreeze_tail(chosen)
        optimizer = torch.optim.Adam(self.get_tail_trainable_params(chosen), lr=lr)
        epoch_losses: list[float] = []
        for _ in range(num_epochs):
            running = 0.0
            for inputs, targets in train_data:
                inputs = _move_nested(inputs, self.device)
                targets = _move_nested(targets, self.device)
                payload = super().edge_forward(inputs, candidate=chosen)
                _, loss = self.cloud_train_step(
                    payload,
                    targets=targets,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    candidate=chosen,
                )
                running += float(loss.item())
            epoch_losses.append(running / max(1, len(train_data)))
        return epoch_losses


def extract_split_features(splitter: UniversalModelSplitter, sample_input: Any) -> SplitPayload | torch.Tensor:
    payload = splitter.edge_forward(sample_input)
    if isinstance(payload, SplitPayload) and len(payload.tensors) == 1:
        return payload.primary_tensor()
    return payload


def save_split_feature_cache(
    cache_path: str,
    frame_index: int,
    intermediate: SplitPayload | torch.Tensor | Mapping[str, torch.Tensor],
    *,
    is_drift: bool = False,
    pseudo_boxes: Any = None,
    pseudo_labels: Any = None,
    pseudo_scores: Any = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> str:
    feature_dir = _cache_feature_dir(cache_path)
    payload = intermediate if isinstance(intermediate, SplitPayload) else (
        SplitPayload.from_mapping(intermediate) if isinstance(intermediate, Mapping)
        else SplitPayload.from_mapping({"payload": intermediate}, primary_label="payload")
    )
    record = {
        "intermediate": payload.cpu(),
        "is_drift": bool(is_drift),
        "pseudo_boxes": pseudo_boxes,
        "pseudo_labels": pseudo_labels,
        "pseudo_scores": pseudo_scores,
        "split_index": payload.split_index,
        "split_label": payload.split_label,
        "candidate_id": payload.candidate_id,
    }
    record.update(dict(extra_metadata or {}))
    out_path = os.path.join(feature_dir, f"{frame_index}.pt")
    torch.save(record, out_path)
    return out_path


def load_split_feature_cache(cache_path: str, frame_index: int) -> dict[str, Any]:
    path = os.path.join(_cache_feature_dir(cache_path), f"{frame_index}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    record = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(record.get("intermediate"), dict):
        record["intermediate"] = SplitPayload.from_mapping(record["intermediate"])
    return record


def universal_split_retrain(
    *,
    model: torch.nn.Module,
    sample_input: Any,
    cache_path: str,
    all_indices: Sequence[int],
    gt_annotations: Mapping[int, Any],
    split_layer: int | None = None,
    candidate_id: str | None = None,
    device: str | torch.device = "cpu",
    num_epoch: int = 1,
    loss_fn=None,
    **_: Any,
) -> list[float]:
    splitter = UniversalModelSplitter(device=device)
    splitter.trace(model, _move_nested(sample_input, splitter.device))
    chosen = splitter.split(candidate_id=candidate_id, layer_index=split_layer)
    splitter.freeze_head(chosen)
    splitter.unfreeze_tail(chosen)
    params = splitter.get_tail_trainable_params(chosen)
    optimizer = torch.optim.Adam(params, lr=1e-3) if params else None
    losses: list[float] = []

    for _ in range(num_epoch):
        epoch_loss = 0.0
        for frame_index in list(all_indices):
            record = load_split_feature_cache(cache_path, frame_index)
            payload = record["intermediate"]
            cached_candidate_id = record.get("candidate_id")
            cached_split_index = record.get("split_index")
            if cached_candidate_id and cached_candidate_id != chosen.candidate_id:
                raise RuntimeError(
                    f"Cached candidate_id={cached_candidate_id} does not match selected {chosen.candidate_id}."
                )
            if cached_split_index is not None and chosen.legacy_layer_index is not None and cached_split_index != chosen.legacy_layer_index:
                raise RuntimeError(
                    f"Cached split_index={cached_split_index} does not match selected {chosen.legacy_layer_index}."
                )

            targets = gt_annotations.get(frame_index)
            if targets is None:
                targets = {
                    "boxes": record.get("pseudo_boxes"),
                    "labels": record.get("pseudo_labels"),
                    "scores": record.get("pseudo_scores"),
                }
            _, loss = splitter.cloud_train_step(
                payload,
                targets=targets,
                loss_fn=loss_fn,
                optimizer=optimizer,
                candidate=chosen,
            )
            epoch_loss += float(loss.item())
        losses.append(epoch_loss / max(1, len(all_indices)))
    return losses


__all__ = [
    "_HAS_TORCHLENS",
    "CandidateProfile",
    "LayerInfo",
    "LayerProfile",
    "SplitCandidate",
    "SplitCandidateSelector",
    "SplitPayload",
    "SplitPointSelector",
    "UniversalModelSplitter",
    "extract_split_features",
    "load_split_feature_cache",
    "save_split_feature_cache",
    "universal_split_retrain",
]
