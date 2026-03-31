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
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from loguru import logger

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


def _split_meta_path(cache_path: str) -> str:
    return os.path.join(_cache_feature_dir(cache_path), "split_meta.json")


def _write_split_cache_metadata(
    cache_path: str,
    payload: SplitPayload,
    extra_metadata: Mapping[str, Any] | None = None,
) -> None:
    if payload.candidate_id is None and payload.split_index is None and payload.split_label is None:
        return
    meta_path = _split_meta_path(cache_path)
    metadata = {
        "universal": True,
        "candidate_id": payload.candidate_id,
        "split_index": payload.split_index,
        "split_label": payload.split_label,
        "boundary_tensor_labels": list(payload.boundary_tensor_labels),
    }
    if extra_metadata:
        metadata.update(dict(extra_metadata))
    if os.path.exists(meta_path):
        try:
            existing = _load_split_cache_metadata(cache_path)
            if existing:
                metadata = {**existing, **metadata}
        except Exception:
            pass
    import json

    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle)


def _load_split_cache_metadata(cache_path: str) -> dict[str, Any]:
    meta_path = _split_meta_path(cache_path)
    if not os.path.exists(meta_path):
        return {}
    try:
        metadata = torch.load(meta_path, map_location="cpu", weights_only=False)
        return metadata if isinstance(metadata, dict) else {}
    except Exception:
        try:
            import json

            with open(meta_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            return loaded if isinstance(loaded, dict) else {}
        except Exception:
            return {}


def _infer_cached_split_choice(
    cache_path: str,
    all_indices: Sequence[int],
) -> tuple[str | None, int | None, list[str] | None]:
    metadata = _load_split_cache_metadata(cache_path)
    candidate_id = metadata.get("candidate_id")
    split_index = metadata.get("split_index")
    boundary_labels = metadata.get("boundary_tensor_labels")
    if candidate_id is not None or split_index is not None:
        return candidate_id, split_index, list(boundary_labels) if boundary_labels else None

    for frame_index in list(all_indices):
        try:
            record = load_split_feature_cache(cache_path, frame_index)
        except FileNotFoundError:
            continue
        payload = record.get("intermediate")
        if isinstance(payload, SplitPayload):
            return payload.candidate_id, payload.split_index, list(payload.boundary_tensor_labels)
        record_boundary = record.get("boundary_tensor_labels")
        return (
            record.get("candidate_id"),
            record.get("split_index"),
            list(record_boundary) if record_boundary else None,
        )
    return None, None, None


class UniversalModelSplitter(GraphSplitRuntime):
    def __init__(self, *, device: str | torch.device = "cpu") -> None:
        super().__init__(device=device)
        self.selected_candidate_id: str | None = None
        self._candidate_enumeration_config: tuple[int, int, int] | None = None

    def trace(
        self,
        model: torch.nn.Module,
        sample_input: Any,
        sample_kwargs: Mapping[str, Any] | None = None,
    ) -> "UniversalModelSplitter":
        super().trace(model, sample_input, sample_kwargs=sample_kwargs)
        self._candidate_enumeration_config = None
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
        self._candidate_enumeration_config = (
            int(max_candidates),
            int(max_boundary_count),
            int(max_payload_bytes),
        )
        return list(self.candidates)

    def sample_candidates(
        self,
        *,
        sample_size: int = 3,
        seed: int = 0,
        require_validated: bool = True,
        require_trainable: bool = False,
        max_candidates: int | None = None,
        max_boundary_count: int = 8,
        max_payload_bytes: int = 32 * 1024 * 1024,
    ) -> list[SplitCandidate]:
        """Return a deterministic random sample of graph-cut candidates.

        This is intended for smoke/regression runs that should exercise several
        different graph partitions instead of always testing the same frontier.
        When validation is requested, only replay-correct candidates are
        sampled. Trainable sampling additionally filters to candidates whose
        cloud tail can receive gradients during validation.
        """

        if not self.candidates:
            self.enumerate_candidates(
                max_candidates=max_candidates or 24,
                max_boundary_count=max_boundary_count,
                max_payload_bytes=max_payload_bytes,
            )

        pool = list(self.candidates[:max_candidates] if max_candidates is not None else self.candidates)
        if require_validated:
            validated: list[SplitCandidate] = []
            for candidate in pool:
                report = self.validate_candidate(candidate)
                if not report["success"]:
                    continue
                if require_trainable and not report["tail_trainability"]:
                    continue
                validated.append(candidate)
            pool = validated
        elif require_trainable:
            pool = [candidate for candidate in pool if candidate.is_trainable_tail]

        if not pool:
            return []

        rng = random.Random(seed)
        if sample_size >= len(pool):
            sampled = list(pool)
            rng.shuffle(sampled)
            return sampled
        return rng.sample(pool, sample_size)

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
        boundary_tensor_labels: Sequence[str] | None = None,
        candidate: SplitCandidate | None = None,
    ) -> SplitCandidate:
        if candidate is not None:
            self.current_candidate = candidate
            self.selected_candidate_id = candidate.candidate_id
            return candidate

        if candidate_id is not None:
            self.current_candidate = self._candidate_or_default(candidate_id)
        elif boundary_tensor_labels is not None:
            boundary_list = list(boundary_tensor_labels)
            boundary_set = set(boundary_list)
            matches = [
                item
                for item in self.candidates
                if item.boundary_tensor_labels == boundary_list
                or set(item.boundary_tensor_labels) == boundary_set
            ]
            if not matches:
                raise KeyError(f"No candidate matches boundary tensor labels {boundary_list!r}.")
            self.current_candidate = matches[0]
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

    def replay_inference(
        self,
        *args: Any,
        candidate: SplitCandidate | str | None = None,
        return_split_output: bool = False,
        **kwargs: Any,
    ) -> Any:
        chosen = self._candidate_or_default(candidate)
        payload = super().edge_forward(*args, candidate=chosen, **kwargs)
        replayed = super().cloud_forward(payload.detach(), *args, candidate=chosen, **kwargs)
        if return_split_output:
            return replayed, payload
        return replayed

    def find_best_split_by_module(self, module_name: str) -> int:
        _, graph = self._ensure_ready()
        module_name = module_name.strip()
        if not module_name:
            raise ValueError("module_name must be non-empty.")
        for candidate in self.candidates:
            for label in candidate.edge_nodes:
                node = graph.nodes[label]
                if node.containing_module and module_name in node.containing_module:
                    return candidate.legacy_layer_index if candidate.legacy_layer_index is not None else 0
        raise KeyError(f"No candidate matched module name {module_name!r}.")

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
    cached_payload = payload.detach().cpu()
    record = {
        "intermediate": cached_payload,
        "is_drift": bool(is_drift),
        "pseudo_boxes": pseudo_boxes,
        "pseudo_labels": pseudo_labels,
        "pseudo_scores": pseudo_scores,
        "split_index": cached_payload.split_index,
        "split_label": cached_payload.split_label,
        "candidate_id": cached_payload.candidate_id,
    }
    record.update(dict(extra_metadata or {}))
    out_path = os.path.join(feature_dir, f"{frame_index}.pt")
    torch.save(record, out_path)
    _write_split_cache_metadata(cache_path, cached_payload, extra_metadata=extra_metadata)
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
    learning_rate: float = 1e-3,
    loss_fn=None,
    splitter: UniversalModelSplitter | None = None,
    chosen_candidate: SplitCandidate | None = None,
    **_: Any,
) -> list[float]:
    splitter = splitter or UniversalModelSplitter(device=device)
    if splitter.graph is None or splitter.model is None:
        splitter.trace(model, _move_nested(sample_input, splitter.device))

    if chosen_candidate is not None:
        chosen = splitter.split(candidate=chosen_candidate)
    else:
        boundary_labels: list[str] | None = None
        if candidate_id is None and split_layer is None:
            candidate_id, split_layer, boundary_labels = _infer_cached_split_choice(cache_path, all_indices)
        if boundary_labels is not None or candidate_id is not None:
            splitter.enumerate_candidates(max_candidates=512)
        if boundary_labels:
            try:
                chosen = splitter.split(boundary_tensor_labels=boundary_labels)
            except KeyError:
                if candidate_id is not None:
                    try:
                        chosen = splitter.split(candidate_id=candidate_id)
                    except KeyError:
                        chosen = splitter.split(layer_index=split_layer)
                else:
                    chosen = splitter.split(layer_index=split_layer)
        else:
            if candidate_id is not None:
                try:
                    chosen = splitter.split(candidate_id=candidate_id)
                except KeyError:
                    chosen = splitter.split(layer_index=split_layer)
            else:
                chosen = splitter.split(layer_index=split_layer)
    splitter.freeze_head(chosen)
    splitter.unfreeze_tail(chosen)
    params = splitter.get_tail_trainable_params(chosen)
    optimizer = torch.optim.Adam(params, lr=float(learning_rate)) if params else None
    losses: list[float] = []
    finite_steps = 0

    for _ in range(num_epoch):
        epoch_loss = 0.0
        for frame_index in list(all_indices):
            record = load_split_feature_cache(cache_path, frame_index)
            payload = record["intermediate"]
            cached_candidate_id = record.get("candidate_id")
            cached_split_index = record.get("split_index")
            payload_boundary_labels = None
            if isinstance(payload, SplitPayload):
                payload_boundary_labels = list(payload.boundary_tensor_labels)
            if payload_boundary_labels is None:
                record_boundary = record.get("boundary_tensor_labels")
                payload_boundary_labels = list(record_boundary) if record_boundary else None
            if payload_boundary_labels and list(chosen.boundary_tensor_labels) != payload_boundary_labels:
                if (
                    isinstance(payload, SplitPayload)
                    and len(payload.tensors) == len(chosen.boundary_tensor_labels)
                ):
                    logger.warning(
                        "Relabelling cached boundary tensors to match the selected candidate. "
                        "cached={} selected={} cached_candidate_id={} selected_candidate_id={} "
                        "cached_split_index={} selected_split_index={}.",
                        payload_boundary_labels,
                        chosen.boundary_tensor_labels,
                        cached_candidate_id,
                        chosen.candidate_id,
                        cached_split_index,
                        chosen.legacy_layer_index,
                    )
                    payload = SplitPayload(
                        tensors=OrderedDict(
                            (label, tensor)
                            for label, tensor in zip(
                                chosen.boundary_tensor_labels,
                                payload.tensors.values(),
                            )
                        ),
                        metadata=dict(payload.metadata),
                        candidate_id=chosen.candidate_id,
                        boundary_tensor_labels=list(chosen.boundary_tensor_labels),
                        primary_label=(
                            chosen.boundary_tensor_labels[-1]
                            if chosen.boundary_tensor_labels
                            else None
                        ),
                        split_index=chosen.legacy_layer_index,
                        split_label=(
                            chosen.boundary_tensor_labels[-1]
                            if chosen.boundary_tensor_labels
                            else None
                        ),
                    )
                else:
                    raise RuntimeError(
                        "Cached boundary tensor labels do not match the selected candidate. "
                        f"cached={payload_boundary_labels}, selected={chosen.boundary_tensor_labels}, "
                        f"cached_candidate_id={cached_candidate_id}, selected_candidate_id={chosen.candidate_id}, "
                        f"cached_split_index={cached_split_index}, selected_split_index={chosen.legacy_layer_index}."
                    )
            if (
                payload_boundary_labels is None
                and cached_candidate_id
                and cached_candidate_id != chosen.candidate_id
                and (
                    cached_split_index is None
                    or chosen.legacy_layer_index is None
                    or cached_split_index != chosen.legacy_layer_index
                )
            ):
                raise RuntimeError(
                    f"Cached candidate_id={cached_candidate_id} does not match selected {chosen.candidate_id}."
                )
            if (
                payload_boundary_labels is None
                and cached_split_index is not None
                and chosen.legacy_layer_index is not None
                and cached_split_index != chosen.legacy_layer_index
            ):
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
            if isinstance(targets, dict):
                split_meta = {
                    "candidate_id": record.get("candidate_id"),
                    "split_index": record.get("split_index"),
                    "split_label": record.get("split_label"),
                    "boundary_tensor_labels": record.get("boundary_tensor_labels"),
                    "sample_id": record.get("sample_id"),
                    "confidence_bucket": record.get("confidence_bucket"),
                    "input_image_size": record.get("input_image_size"),
                    "input_tensor_shape": record.get("input_tensor_shape"),
                }
                targets = {
                    **targets,
                    "_split_meta": split_meta,
                }
            _, loss = splitter.cloud_train_step(
                payload,
                targets=targets,
                loss_fn=loss_fn,
                optimizer=optimizer,
                candidate=chosen,
            )
            if not bool(torch.isfinite(loss).item()):
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                continue
            finite_steps += 1
            epoch_loss += float(loss.item())
        losses.append(epoch_loss / max(1, len(all_indices)))
    if optimizer is not None and params and finite_steps == 0:
        raise RuntimeError(
            "Split retraining did not produce any finite optimization step. "
            "The selected candidate may replay empty/non-differentiable detector outputs."
        )
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
