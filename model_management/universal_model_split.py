"""
Graph-based split runtime for arbitrary TorchLens-traceable PyTorch models.

The public facade remains `UniversalModelSplitter`, but the implementation is
candidate-driven:
  - a split is a validated graph partition candidate
  - replay is dependency-driven over a graph IR
  - payloads are minimal boundary tensor maps
"""

from __future__ import annotations

import hashlib
import os
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from loguru import logger

from model_management.activation_sparsity import apply_das_to_model, apply_das_to_tail
from model_management.candidate_generator import (
    build_candidate_from_edge_seed,
    enumerate_candidates,
    estimate_privacy_leakage_from_edge_params,
)
from model_management.candidate_profiler import profile_candidates
from model_management.candidate_selector import SplitCandidateSelector, SplitPointSelector
from model_management.payload import SplitPayload
from model_management.split_candidate import CandidateProfile, SplitCandidate
from model_management.split_runtime import GraphSplitRuntime, _call_loss_fn


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


def _batch_tensors(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        raise ValueError("Cannot batch an empty tensor sequence.")
    first = tensors[0]
    
    if first.ndim == 0:
        return torch.stack(tensors, dim=0)

    # Note: Edge execution usually uses batch_size=1, producing tensors like [1, C, H, W].
    # If first dim is 1, `cat` restores [B, C, H, W]. 
    # If first dim > 1 but shapes vary (e.g. variable bounding boxes), we must use `cat`.
    # If first dim > 1 and shapes are identical, it might be an unbatched feature [C, H, W], 
    # so we should `stack` to get [B, C, H, W].
    shapes_match = all(t.shape == first.shape for t in tensors)
    if shapes_match:
        # If the shape has a batch dim > 1, stacking would make 5D. 
        # Safest default for batched inputs vs single inputs
        if first.ndim > 0:
            return torch.cat(tensors, dim=0)
        return torch.stack(tensors, dim=0)
    else:
        # Variable shapes, e.g. [N_i, 6] boxes per image.
        # Note: torch.cat works, but for sequence tasks or detection targets if
        # shapes completely mismatch returning a list or handling padding could be needed.
        # But since we use `torch.cat`, we assume the tensors combine along dim 0 correctly.
        return torch.cat(tensors, dim=0)


def _batch_payloads(
    payloads: Sequence[SplitPayload | torch.Tensor],
    *,
    boundary_tensor_labels: Sequence[str] | None = None,
    candidate_id: str | None = None,
) -> SplitPayload | torch.Tensor:
    if not payloads:
        raise ValueError("Cannot batch an empty payload sequence.")
    first = payloads[0]
    if isinstance(first, SplitPayload):
        if not all(isinstance(payload, SplitPayload) for payload in payloads):
            raise TypeError("Cannot mix SplitPayload and Tensor payloads when batching.")
        labels = list(boundary_tensor_labels or first.boundary_tensor_labels)
        batched_tensors = OrderedDict()
        for label in labels:
            batched_tensors[label] = _batch_tensors([payload.tensors[label] for payload in payloads])
        return SplitPayload(
            tensors=batched_tensors,
            metadata=first.metadata,
            candidate_id=candidate_id if candidate_id is not None else first.candidate_id,
            boundary_tensor_labels=labels,
            primary_label=first.primary_label,
            split_index=first.split_index,
            split_label=first.split_label,
        )
    if isinstance(first, torch.Tensor):
        if not all(isinstance(payload, torch.Tensor) for payload in payloads):
            raise TypeError("Cannot mix SplitPayload and Tensor payloads when batching.")
        return _batch_tensors([payload for payload in payloads])
    raise TypeError(f"Unsupported payload type for batching: {type(first)!r}")


def _cache_feature_dir(cache_path: str) -> str:
    feature_dir = os.path.join(cache_path, "features")
    os.makedirs(feature_dir, exist_ok=True)
    return feature_dir


def _split_meta_path(cache_path: str) -> str:
    return os.path.join(_cache_feature_dir(cache_path), "split_meta.json")


def _format_boundary_labels(
    boundary_tensor_labels: Sequence[str] | None,
    *,
    max_items: int = 4,
) -> str:
    labels = [str(label) for label in (boundary_tensor_labels or ())]
    if not labels:
        return "[]"
    if len(labels) <= max_items:
        return "[" + ", ".join(labels) + "]"
    shown = ", ".join(labels[:max_items])
    return f"[{shown}, ... (+{len(labels) - max_items} more)]"


def _describe_split_identity(
    *,
    candidate_id: str | None,
    boundary_tensor_labels: Sequence[str] | None,
    legacy_split_index: int | None,
) -> str:
    labels = list(boundary_tensor_labels or [])
    return (
        f"candidate_id={candidate_id}, "
        f"boundary_count={len(labels)}, "
        f"boundary_tensor_labels={_format_boundary_labels(labels)}, "
        f"legacy_split_index={legacy_split_index}"
    )


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

    def _candidate_from_boundary_labels(self, boundary_tensor_labels: Sequence[str]) -> SplitCandidate:
        _, graph = self._ensure_ready()
        boundary_list = list(boundary_tensor_labels)
        raw = ",".join(boundary_list).encode("utf-8")
        candidate = build_candidate_from_edge_seed(
            graph,
            candidate_id=f"boundary_{hashlib.sha1(raw).hexdigest()[:12]}",
            edge_seed_nodes=boundary_list,
            metadata={"source": "boundary_tensor_labels"},
        )
        if candidate is None:
            raise KeyError(f"No candidate matches boundary tensor labels {boundary_list!r}.")
        if set(candidate.boundary_tensor_labels) != set(boundary_list):
            raise KeyError(f"No candidate matches boundary tensor labels {boundary_list!r}.")
        if all(item.candidate_id != candidate.candidate_id for item in self.candidates):
            self.candidates.append(candidate)
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
            self.current_candidate = matches[0] if matches else self._candidate_from_boundary_labels(boundary_list)
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
        cumulative_params = 0
        seen_parameters: set[str] = set()
        parameter_numels = dict(getattr(graph, "parameter_numels", {}) or {})
        for index, label in enumerate(graph.relevant_labels):
            node = graph.nodes[label]
            cumulative_flops += node.estimated_flops
            for ref in node.parameter_refs:
                if ref.fq_name in seen_parameters:
                    continue
                seen_parameters.add(ref.fq_name)
                cumulative_params += int(parameter_numels.get(ref.fq_name, 0))
            privacy = estimate_privacy_leakage_from_edge_params(cumulative_params)
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
        batch_size: int = 1,
        lr: float = 1e-3,
        candidate: SplitCandidate | str | None = None,
    ) -> list[float]:
        chosen = self._candidate_or_default(candidate)
        self.freeze_head(chosen)
        self.unfreeze_tail(chosen)
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        optimizer = torch.optim.Adam(self.get_tail_trainable_params(chosen), lr=lr)
        epoch_losses: list[float] = []
        for _ in range(num_epochs):
            running = 0.0
            seen = 0
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i : i + batch_size]
                payloads: list[SplitPayload | torch.Tensor] = []
                targets_list: list[Any] = []
                for inputs, targets in batch:
                    inputs = _move_nested(inputs, self.device)
                    targets = _move_nested(targets, self.device)
                    payloads.append(super().edge_forward(inputs, candidate=chosen))
                    targets_list.append(targets)

                if not payloads:
                    continue

                batched_payload = _batch_payloads(
                    payloads,
                    boundary_tensor_labels=chosen.boundary_tensor_labels,
                    candidate_id=chosen.candidate_id,
                )
                batched_targets = targets_list[0] if len(targets_list) == 1 else targets_list
                _, loss = self.cloud_train_step(
                    batched_payload,
                    targets=batched_targets,
                    loss_fn=loss_fn,
                    optimizer=None,
                    candidate=chosen,
                )
                if bool(torch.isfinite(loss).item()):
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    self._invalidate_validation_cache()
                    batch_samples = len(batch)
                    running += float(loss.item()) * batch_samples
                    seen += batch_samples
            epoch_losses.append(running / max(1, seen))
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
    batch_size: int = 1,
    learning_rate: float = 1e-3,
    loss_fn=None,
    splitter: UniversalModelSplitter | None = None,
    chosen_candidate: SplitCandidate | None = None,
    das_enabled: bool = False,
    das_bn_only: bool = False,
    das_probe_samples: int = 10,
    das_strategy: str = "tgi",
    das_use_spectral_entropy: bool = False,
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
    das_strategy = str(das_strategy).strip().lower()
    if das_use_spectral_entropy:
        das_strategy = "entropy"
    if das_strategy not in {"tgi", "entropy"}:
        raise ValueError(f"Unsupported DAS strategy: {das_strategy!r}")
    das_use_entropy = das_strategy == "entropy"

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

    if num_epoch > 1:
        logger.info("[FixedSplitCL] Executing universal_split_retrain: num_epoch={}, batch_size={}, samples={}.", num_epoch, batch_size, len(all_indices))

    def _infer_das_tail_modules() -> list[str]:
        _, graph = splitter._ensure_ready()
        module_names: set[str] = set()
        for label in chosen.cloud_nodes:
            node = graph.nodes.get(label)
            if node is None:
                continue
            if node.containing_module:
                parent = node.containing_module.rsplit(".", 1)[0] if "." in node.containing_module else node.containing_module
                if parent:
                    module_names.add(parent)
            for ref in node.parameter_refs:
                fq_name = getattr(ref, "fq_name", "")
                module_path = fq_name.rsplit(".", 1)[0] if "." in fq_name else ""
                if module_path:
                    parent = module_path.rsplit(".", 1)[0] if "." in module_path else module_path
                    if parent:
                        module_names.add(parent)
        return sorted(module_names)

    das_trainer = None
    if das_enabled:
        tail_modules = _infer_das_tail_modules()
        if tail_modules:
            das_trainer = apply_das_to_tail(
                model,
                tail_modules,
                bn_only=das_bn_only,
                strategy=das_strategy,
                use_spectral_entropy=das_use_entropy,
                device=device,
            )
        else:
            das_trainer = apply_das_to_model(
                model,
                bn_only=das_bn_only,
                probe_samples=das_probe_samples,
                strategy=das_strategy,
                use_spectral_entropy=das_use_entropy,
                device=device,
            )
        das_trainer.probe_samples = max(0, int(das_probe_samples)) if das_use_entropy else max(1, int(das_probe_samples))

    params = splitter.get_tail_trainable_params(chosen)
    optimizer = torch.optim.Adam(params, lr=float(learning_rate)) if params else None
    losses: list[float] = []
    finite_steps = 0

    for _epoch in range(num_epoch):
        epoch_loss = 0.0
        epoch_finite_samples = 0
        if das_trainer is not None and not das_use_entropy:
            probe_count = min(len(all_indices), int(das_trainer.probe_samples))
            probe_indices = (
                random.sample(list(all_indices), probe_count)
                if probe_count and len(all_indices) > probe_count
                else list(all_indices)[:probe_count]
            )
            aggregated: dict[str, float] = {}
            counts: dict[str, int] = {}
            for probe_index in probe_indices:
                record = load_split_feature_cache(cache_path, probe_index)
                payload = record["intermediate"]
                targets = gt_annotations.get(probe_index)
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

                def _probe_forward():
                    outputs = splitter.cloud_forward(payload, candidate=chosen)
                    effective_loss_fn = loss_fn if loss_fn is not None else splitter.trainability_loss_fn
                    loss = _call_loss_fn(
                        effective_loss_fn,
                        outputs,
                        targets,
                        runtime=splitter,
                        candidate=chosen,
                    )
                    return {"loss": loss}

                ratios = das_trainer.probe_with_targets(_probe_forward)
                for key, value in ratios.items():
                    aggregated[key] = aggregated.get(key, 0.0) + float(value)
                    counts[key] = counts.get(key, 0) + 1

            if counts:
                averaged = {key: aggregated[key] / counts[key] for key in aggregated}
            else:
                averaged = {}
            das_trainer.activate_sparsity(averaged)
        elif das_trainer is not None and das_use_entropy:
            # Mirror the gradient-based flow: probe first, then keep ratios fixed
            # for the remainder of the epoch.
            probe_count = min(len(all_indices), int(getattr(das_trainer, "probe_samples", 0) or 0))
            if probe_count <= 0:
                das_trainer.activate_sparsity({})
            else:
                probe_indices = (
                    random.sample(list(all_indices), probe_count)
                    if probe_count and len(all_indices) > probe_count
                    else list(all_indices)[:probe_count]
                )
                das_trainer.deactivate_sparsity()
                aggregated: dict[str, float] = {}
                counts: dict[str, int] = {}
                with torch.no_grad():
                    for probe_index in probe_indices:
                        record = load_split_feature_cache(cache_path, probe_index)
                        payload = record["intermediate"]
                        splitter.cloud_forward(payload, candidate=chosen)
                        ratios = das_trainer.refresh_pruning_ratios_from_entropy()
                        for key, value in ratios.items():
                            aggregated[key] = aggregated.get(key, 0.0) + float(value)
                            counts[key] = counts.get(key, 0) + 1
                averaged = {key: aggregated[key] / counts[key] for key in aggregated} if counts else {}
                das_trainer.activate_sparsity(averaged)

        for i in range(0, len(all_indices), batch_size):
            batch_indices = all_indices[i:i + batch_size]
            batch_loss = 0.0
            
            payloads = []
            targets_list = []
            
            for frame_index in batch_indices:
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
                            "Relabelling cached boundary tensors to match the selected split combination. "
                            "cached=({}) selected=({}).",
                            _describe_split_identity(
                                candidate_id=cached_candidate_id,
                                boundary_tensor_labels=payload_boundary_labels,
                                legacy_split_index=cached_split_index,
                            ),
                            _describe_split_identity(
                                candidate_id=chosen.candidate_id,
                                boundary_tensor_labels=chosen.boundary_tensor_labels,
                                legacy_split_index=chosen.legacy_layer_index,
                            ),
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
                            "Cached boundary tensor labels do not match the selected split combination. "
                            f"cached=({_describe_split_identity(candidate_id=cached_candidate_id, boundary_tensor_labels=payload_boundary_labels, legacy_split_index=cached_split_index)}), "
                            f"selected=({_describe_split_identity(candidate_id=chosen.candidate_id, boundary_tensor_labels=chosen.boundary_tensor_labels, legacy_split_index=chosen.legacy_layer_index)})."
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
                        "Cached split identity does not match the selected split combination. "
                        f"cached=({_describe_split_identity(candidate_id=cached_candidate_id, boundary_tensor_labels=None, legacy_split_index=cached_split_index)}), "
                        f"selected=({_describe_split_identity(candidate_id=chosen.candidate_id, boundary_tensor_labels=chosen.boundary_tensor_labels, legacy_split_index=chosen.legacy_layer_index)})."
                    )
                if (
                    payload_boundary_labels is None
                    and cached_split_index is not None
                    and chosen.legacy_layer_index is not None
                    and cached_split_index != chosen.legacy_layer_index
                ):
                    raise RuntimeError(
                        "Cached legacy split index does not match the selected split combination. "
                        f"cached=({_describe_split_identity(candidate_id=cached_candidate_id, boundary_tensor_labels=None, legacy_split_index=cached_split_index)}), "
                        f"selected=({_describe_split_identity(candidate_id=chosen.candidate_id, boundary_tensor_labels=chosen.boundary_tensor_labels, legacy_split_index=chosen.legacy_layer_index)})."
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
                        "input_resize_mode": record.get("input_resize_mode"),
                    }
                    targets = {
                        **targets,
                        "_split_meta": split_meta,
                    }
                
                payloads.append(payload)
                targets_list.append(targets)

            if not payloads:
                continue

            batched_payload = _batch_payloads(
                payloads,
                boundary_tensor_labels=chosen.boundary_tensor_labels,
                candidate_id=chosen.candidate_id,
            )

            collate_targets = targets_list

            _, loss = splitter.cloud_train_step(
                batched_payload,
                targets=collate_targets,
                loss_fn=loss_fn,
                optimizer=None, # accumulation is no longer needed since it's batched
                candidate=chosen,
            )

            if bool(torch.isfinite(loss).item()):
                batch_loss = loss
            else:
                batch_loss = 0.0

            if not isinstance(batch_loss, float): # If we actually computed valid losses
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    batch_loss.backward()
                    optimizer.step()
                    splitter._invalidate_validation_cache()
                finite_steps += len(batch_indices)
                epoch_finite_samples += len(batch_indices)
                epoch_loss += float(batch_loss.item()) * len(batch_indices)
        losses.append(epoch_loss / max(1, epoch_finite_samples))
        batch_count = (len(all_indices) + batch_size - 1) // batch_size
        if num_epoch > 1 or _epoch == num_epoch - 1:
            logger.info(
                "[FixedSplitCL] epoch {}/{} finished. Samples: {}, Batch Size: {}, Batches: {}, Avg loss: {:.4f}", 
                _epoch + 1, num_epoch, len(all_indices), batch_size, batch_count, losses[-1]
            )
    if optimizer is not None and params and finite_steps == 0:
        raise RuntimeError(
            "Split retraining did not produce any finite optimization step. "
            "The selected candidate may replay empty/non-differentiable detector outputs."
        )
    return losses


__all__ = [
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
