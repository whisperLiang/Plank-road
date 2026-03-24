from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch

from model_management.candidate_generator import enumerate_candidates
from model_management.graph_ir import (
    GraphIR,
    build_graph_from_trace,
    flatten_bound_inputs,
    materialize_tree_spec,
    normalize_runtime_inputs,
    _resolve_attr_path,
    trace_model,
)
from model_management.payload import SplitPayload
from model_management.split_candidate import SplitCandidate


def _flatten_tensors(obj: Any) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    if isinstance(obj, torch.Tensor):
        tensors.append(obj)
        return tensors
    if isinstance(obj, list):
        for item in obj:
            tensors.extend(_flatten_tensors(item))
        return tensors
    if isinstance(obj, tuple):
        for item in obj:
            tensors.extend(_flatten_tensors(item))
        return tensors
    if isinstance(obj, dict):
        for value in obj.values():
            tensors.extend(_flatten_tensors(value))
    return tensors


def _coerce_numeric_tensor(obj: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point():
            return obj.to(device=device, dtype=dtype)
        if obj.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
            return obj.to(device=device, dtype=dtype)
        return None
    if isinstance(obj, (list, tuple)):
        try:
            tensor = torch.as_tensor(obj, device=device, dtype=dtype)
        except Exception:
            return None
        return tensor
    return None


def _aligned_tensor_loss(output_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor | None:
    if not output_tensor.is_floating_point():
        return None
    output_flat = output_tensor.reshape(-1)
    target_flat = target_tensor.reshape(-1)
    count = min(output_flat.numel(), target_flat.numel())
    loss = output_flat.new_zeros(())
    pieces = 0
    if count > 0:
        lhs = output_flat[:count]
        rhs = target_flat[:count]
        finite_mask = torch.isfinite(lhs) & torch.isfinite(rhs)
        if torch.any(finite_mask):
            loss = loss + torch.nn.functional.mse_loss(lhs[finite_mask], rhs[finite_mask])
            pieces += 1
    if output_flat.numel() > count:
        tail = output_flat[count:]
        finite_mask = torch.isfinite(tail)
        if torch.any(finite_mask):
            valid_tail = tail[finite_mask]
            loss = loss + torch.nn.functional.mse_loss(
                valid_tail,
                torch.zeros_like(valid_tail),
            )
            pieces += 1
    if pieces == 0:
        return None
    return loss / pieces


def _structured_supervision_loss(outputs: Any, targets: Any) -> torch.Tensor | None:
    if targets is None:
        return None

    if isinstance(outputs, (list, tuple)) and len(outputs) == 1 and isinstance(targets, Mapping):
        return _structured_supervision_loss(outputs[0], targets)

    if isinstance(outputs, Mapping) and isinstance(targets, (list, tuple)) and len(targets) == 1:
        return _structured_supervision_loss(outputs, targets[0])

    if isinstance(outputs, torch.Tensor):
        target_tensor = _coerce_numeric_tensor(
            targets,
            device=outputs.device,
            dtype=outputs.dtype if outputs.is_floating_point() else torch.float32,
        )
        if target_tensor is None:
            return None
        return _aligned_tensor_loss(outputs, target_tensor)

    if isinstance(outputs, dict) and isinstance(targets, Mapping):
        total: torch.Tensor | None = None
        pieces = 0
        for key in outputs.keys() & targets.keys():
            partial = _structured_supervision_loss(outputs[key], targets[key])
            if partial is None:
                continue
            total = partial if total is None else total + partial
            pieces += 1
        if total is None:
            return None
        return total / max(1, pieces)

    if isinstance(outputs, (list, tuple)) and isinstance(targets, (list, tuple)):
        total: torch.Tensor | None = None
        pieces = 0
        for lhs, rhs in zip(outputs, targets):
            partial = _structured_supervision_loss(lhs, rhs)
            if partial is None:
                continue
            total = partial if total is None else total + partial
            pieces += 1
        if total is None:
            return None
        return total / max(1, pieces)

    return None


def reduce_output_to_loss(outputs: Any, targets: Any = None) -> torch.Tensor:
    supervised = _structured_supervision_loss(outputs, targets)
    if supervised is not None:
        return supervised
    accumulator: torch.Tensor | None = None
    anchor: torch.Tensor | None = None
    for tensor in _flatten_tensors(outputs):
        if not tensor.is_floating_point():
            continue
        if anchor is None:
            anchor = tensor
        if tensor.numel() == 0:
            continue
        finite_mask = torch.isfinite(tensor)
        if not torch.any(finite_mask):
            continue
        value = tensor[finite_mask].mean()
        accumulator = value if accumulator is None else accumulator + value
    if accumulator is None:
        if anchor is not None:
            return anchor.sum() * 0.0
        raise RuntimeError("Could not reduce structured output to a differentiable scalar.")
    return accumulator


def compare_outputs(expected: Any, replayed: Any, *, atol: float = 1e-4, rtol: float = 1e-4) -> tuple[bool, float]:
    expected_tensors = _flatten_tensors(expected)
    replayed_tensors = _flatten_tensors(replayed)
    if len(expected_tensors) != len(replayed_tensors):
        return False, float("inf")
    max_diff = 0.0
    for lhs, rhs in zip(expected_tensors, replayed_tensors):
        if lhs.shape != rhs.shape:
            return False, float("inf")
        if lhs.numel() == 0 and rhs.numel() == 0:
            continue
        lhs_cpu = lhs.detach().cpu()
        rhs_cpu = rhs.detach().cpu()
        if lhs_cpu.dtype == torch.bool and rhs_cpu.dtype == torch.bool:
            diff = float(torch.count_nonzero(lhs_cpu != rhs_cpu).item())
        else:
            diff = float((lhs_cpu - rhs_cpu).abs().max().item())
        max_diff = max(max_diff, diff)
        if lhs_cpu.dtype == torch.bool and rhs_cpu.dtype == torch.bool:
            if not torch.equal(lhs_cpu, rhs_cpu):
                return False, max_diff
            continue
        if not torch.allclose(lhs_cpu, rhs_cpu, atol=atol, rtol=rtol):
            return False, max_diff
    return True, max_diff


def _maybe_retry_getitem_with_safe_indexing(func: Any, args: list[Any], kwargs: dict[str, Any]) -> Any | None:
    func_name = getattr(func, "__name__", None)
    if func_name != "__getitem__":
        return None
    if kwargs or len(args) < 2:
        return None
    source, index = args[0], args[1]
    if not isinstance(source, torch.Tensor) or source.ndim == 0:
        return None
    if not isinstance(index, torch.Tensor):
        return None
    if index.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        return None
    dim0 = source.shape[0]
    safe_index = index[(index >= -dim0) & (index < dim0)]
    return func(source, safe_index)


@dataclass
class RuntimeState:
    values: "OrderedDict[str, Any]"


class GraphSplitRuntime:
    def __init__(self, *, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.model: torch.nn.Module | None = None
        self.graph: GraphIR | None = None
        self.history: Any | None = None
        self.candidates: list[SplitCandidate] = []
        self.current_candidate: SplitCandidate | None = None
        self.runtime_state: RuntimeState = RuntimeState(values=OrderedDict())

    def trace(
        self,
        model: torch.nn.Module,
        sample_input: Any,
        sample_kwargs: Mapping[str, Any] | None = None,
    ) -> "GraphSplitRuntime":
        self.model = model
        history, sample_args, sample_kwargs_dict, sample_output = trace_model(
            model,
            sample_input,
            sample_kwargs=sample_kwargs,
        )
        self.history = history
        self.graph = build_graph_from_trace(
            model,
            history,
            sample_args,
            sample_kwargs_dict,
            sample_output,
        )
        self.candidates = enumerate_candidates(self.graph)
        self.current_candidate = self.candidates[0] if self.candidates else None
        self.reset_runtime_state()
        return self

    def reset_runtime_state(self) -> None:
        self.runtime_state = RuntimeState(values=OrderedDict())

    def _ensure_ready(self) -> tuple[torch.nn.Module, GraphIR]:
        if self.model is None or self.graph is None:
            raise RuntimeError("GraphSplitRuntime.trace() must be called first.")
        return self.model, self.graph

    def materialize_node(
        self,
        label: str,
        available_tensors: Mapping[str, Any],
        *,
        clone_parent_labels: set[str] | None = None,
        clone_cache: dict[tuple[str, tuple[Any, ...] | None], Any] | None = None,
    ) -> Any:
        model, graph = self._ensure_ready()
        node = graph.nodes[label]
        if node.is_input:
            return available_tensors[label]
        if node.node_type == "buffer" and node.buffer_refs:
            buffer_ref = node.buffer_refs[0]
            module = _resolve_attr_path(model, buffer_ref.module_path)
            try:
                return module.get_buffer(buffer_ref.buffer_name)
            except (AttributeError, TypeError):
                if isinstance(module, (list, tuple)) and buffer_ref.buffer_name.isdigit():
                    return module[int(buffer_ref.buffer_name)]
                if isinstance(module, dict):
                    return module[buffer_ref.buffer_name]
                return getattr(module, buffer_ref.buffer_name)
        if node.is_output:
            if not node.parent_labels:
                raise RuntimeError(f"Output node {label} has no parent.")
            return available_tensors[node.parent_labels[0]]
        if node.func is None:
            if node.parent_labels:
                return available_tensors[node.parent_labels[0]]
            raise RuntimeError(f"Node {label} has no executable function.")

        args, kwargs = node.resolve_args(
            available_tensors,
            model,
            device=self.device,
            clone_parent_labels=clone_parent_labels,
            clone_cache=clone_cache,
        )
        try:
            output = node.func(*args, **kwargs)
        except IndexError:
            args_list = list(args) if isinstance(args, tuple) else list(args)
            kwargs_dict = dict(kwargs)
            output = _maybe_retry_getitem_with_safe_indexing(node.func, args_list, kwargs_dict)
            if output is None:
                raise
        if output is None and (node.is_inplace or node.func_name.lower() in {"__setitem__", "setitem", "append", "extend", "update"}):
            output = args[0] if args else None
        if node.is_multi_output and node.multi_output_index is not None:
            if isinstance(output, dict):
                keys = list(output.keys())
                output = output[keys[node.multi_output_index]]
            else:
                output = output[node.multi_output_index]
        return output

    def replay_subgraph(
        self,
        node_labels: Iterable[str],
        initial_tensors: Mapping[str, Any],
    ) -> OrderedDict[str, Any]:
        _, graph = self._ensure_ready()
        node_set = set(node_labels)
        available = OrderedDict((label, tensor) for label, tensor in initial_tensors.items())
        remaining_users: dict[str, int] = {label: 0 for label in node_set}
        for label in graph.topological_order:
            if label not in node_set:
                continue
            for parent in graph.nodes[label].parent_labels:
                if parent in node_set:
                    remaining_users[parent] = remaining_users.get(parent, 0) + 1
        for label in graph.topological_order:
            if label not in node_set or label in available:
                continue
            node = graph.nodes[label]
            missing = [parent for parent in node.parent_labels if parent not in available]
            if missing:
                raise RuntimeError(
                    f"Cannot materialize {label}; missing parent tensors {missing}. "
                    "This indicates an invalid split candidate or incomplete payload."
                )
            clone_parent_labels = {
                parent
                for parent in node.parent_labels
                if node.is_inplace and remaining_users.get(parent, 0) > 1
            }
            available[label] = self.materialize_node(
                label,
                available,
                clone_parent_labels=clone_parent_labels,
                clone_cache={},
            )
            for parent in node.parent_labels:
                if parent in remaining_users:
                    remaining_users[parent] = max(0, remaining_users[parent] - 1)
        return available

    def _bound_inputs_to_labels(
        self,
        *args: Any,
        partial: bool = True,
        **kwargs: Any,
    ) -> OrderedDict[str, torch.Tensor]:
        _, graph = self._ensure_ready()
        bound = normalize_runtime_inputs(graph.forward_signature, *args, partial=partial, **kwargs)
        flattened = flatten_bound_inputs(bound)
        label_map: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for address, tensor in flattened.items():
            label = graph.input_address_to_label.get(address)
            if label is not None:
                label_map[label] = tensor.to(self.device)
        return label_map

    def _candidate_or_default(self, candidate: SplitCandidate | str | None = None) -> SplitCandidate:
        if isinstance(candidate, SplitCandidate):
            return candidate
        if isinstance(candidate, str):
            for item in self.candidates:
                if item.candidate_id == candidate:
                    return item
            raise KeyError(f"Unknown candidate_id: {candidate}")
        if self.current_candidate is None:
            raise RuntimeError("No split candidate is currently selected.")
        return self.current_candidate

    def _payload_from_available(self, candidate: SplitCandidate, available: Mapping[str, Any]) -> SplitPayload:
        tensors = OrderedDict()
        for label in candidate.boundary_tensor_labels:
            value = available[label]
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Boundary label {label} produced {type(value)!r}; "
                    "graph-partition payloads must cross tensor-valued boundaries."
                )
            tensors[label] = value
        primary_label = candidate.boundary_tensor_labels[-1] if candidate.boundary_tensor_labels else None
        return SplitPayload(
            tensors=tensors,
            metadata={"cloud_input_labels": candidate.cloud_input_labels},
            candidate_id=candidate.candidate_id,
            boundary_tensor_labels=list(candidate.boundary_tensor_labels),
            primary_label=primary_label,
            split_index=candidate.legacy_layer_index,
            split_label=primary_label,
        )

    def edge_forward(
        self,
        *args: Any,
        candidate: SplitCandidate | str | None = None,
        **kwargs: Any,
    ) -> SplitPayload:
        chosen = self._candidate_or_default(candidate)
        input_tensors = self._bound_inputs_to_labels(*args, partial=True, **kwargs)
        missing = [label for label in chosen.edge_input_labels if label not in input_tensors]
        if missing:
            raise RuntimeError(f"Missing edge inputs for labels: {missing}")
        self.reset_runtime_state()
        available = self.replay_subgraph(chosen.edge_nodes, input_tensors)
        payload = self._payload_from_available(chosen, available)
        self.runtime_state.values = available
        return payload

    def cloud_forward(
        self,
        payload: SplitPayload | Mapping[str, torch.Tensor] | torch.Tensor,
        *args: Any,
        candidate: SplitCandidate | str | None = None,
        **kwargs: Any,
    ) -> Any:
        _, graph = self._ensure_ready()
        chosen = self._candidate_or_default(candidate)
        self.reset_runtime_state()

        if isinstance(payload, SplitPayload):
            payload_tensors = OrderedDict((label, tensor.to(self.device)) for label, tensor in payload.tensors.items())
        elif isinstance(payload, torch.Tensor):
            if not chosen.boundary_tensor_labels:
                raise RuntimeError("The selected candidate has no boundary tensors.")
            payload_tensors = OrderedDict([(chosen.boundary_tensor_labels[-1], payload.to(self.device))])
        else:
            payload_tensors = OrderedDict((str(label), tensor.to(self.device)) for label, tensor in payload.items())

        cloud_inputs = self._bound_inputs_to_labels(*args, partial=True, **kwargs)
        missing = [
            label
            for label in chosen.cloud_input_labels
            if label not in payload_tensors and label not in cloud_inputs
        ]
        if missing:
            raise RuntimeError(f"Missing cloud-side explicit inputs for labels: {missing}")

        initial = OrderedDict()
        initial.update(payload_tensors)
        initial.update(cloud_inputs)
        available = self.replay_subgraph(chosen.cloud_nodes, initial)
        self.runtime_state.values = available

        output_tensor_map = OrderedDict()
        for address, label in graph.output_address_to_label.items():
            if label in available and isinstance(available[label], torch.Tensor):
                output_tensor_map[address] = available[label]
        return materialize_tree_spec(graph.output_spec, output_tensor_map)

    def full_replay(self, *args: Any, **kwargs: Any) -> Any:
        _, graph = self._ensure_ready()
        inputs = self._bound_inputs_to_labels(*args, partial=False, **kwargs)
        available = self.replay_subgraph(graph.relevant_labels, inputs)
        self.runtime_state.values = available
        output_tensor_map = OrderedDict()
        for address, label in graph.output_address_to_label.items():
            if label in available and isinstance(available[label], torch.Tensor):
                output_tensor_map[address] = available[label]
        return materialize_tree_spec(graph.output_spec, output_tensor_map)

    def validate_candidate(
        self,
        candidate: SplitCandidate | str | None = None,
        *,
        atol: float = 1e-4,
        rtol: float = 1e-4,
    ) -> dict[str, Any]:
        model, graph = self._ensure_ready()
        chosen = self._candidate_or_default(candidate)
        sample_args = tuple(arg.to(self.device) if isinstance(arg, torch.Tensor) else arg for arg in graph.sample_args)
        sample_kwargs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in graph.sample_kwargs.items()
        }

        start = time.perf_counter()
        payload = self.edge_forward(*sample_args, candidate=chosen, **sample_kwargs)
        edge_elapsed = time.perf_counter() - start
        self.reset_runtime_state()

        cloud_start = time.perf_counter()
        replayed = self.cloud_forward(payload.detach(), *sample_args, candidate=chosen, **sample_kwargs)
        cloud_elapsed = time.perf_counter() - cloud_start

        with torch.no_grad():
            expected = model(*sample_args, **sample_kwargs)
        success, max_diff = compare_outputs(expected, replayed, atol=atol, rtol=rtol)

        trainability = chosen.is_trainable_tail
        if success and trainability:
            trainability = self._check_tail_trainability(chosen, payload.detach(), *sample_args, **sample_kwargs)

        chosen.is_validated = success
        chosen.validation_error = None if success else f"max_diff={max_diff}"
        return {
            "success": success,
            "max_diff": max_diff,
            "edge_latency": edge_elapsed,
            "cloud_latency": cloud_elapsed,
            "end_to_end_latency": edge_elapsed + cloud_elapsed,
            "tail_trainability": trainability,
            "stability_score": 1.0 if success else 0.0,
            "error": chosen.validation_error,
        }

    def _check_tail_trainability(
        self,
        candidate: SplitCandidate,
        payload: SplitPayload,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        params = self._candidate_parameters(candidate.cloud_nodes)
        if not params:
            return False
        self._zero_candidate_grads(candidate.cloud_nodes)
        outputs = self.cloud_forward(payload, *args, candidate=candidate, **kwargs)
        loss = reduce_output_to_loss(outputs)
        try:
            loss.backward()
        except Exception:
            self._zero_candidate_grads(candidate.cloud_nodes)
            return False
        has_grad = any(param.grad is not None and torch.count_nonzero(param.grad).item() > 0 for param in params)
        self._zero_candidate_grads(candidate.cloud_nodes)
        return has_grad

    def _candidate_param_names(self, node_labels: Iterable[str]) -> list[str]:
        _, graph = self._ensure_ready()
        names: list[str] = []
        seen: set[str] = set()
        for label in node_labels:
            node = graph.nodes[label]
            for ref in node.parameter_refs:
                if ref.fq_name not in seen:
                    seen.add(ref.fq_name)
                    names.append(ref.fq_name)
        return names

    def _candidate_parameters(self, node_labels: Iterable[str]) -> list[torch.nn.Parameter]:
        model, _ = self._ensure_ready()
        parameters: list[torch.nn.Parameter] = []
        for fq_name in self._candidate_param_names(node_labels):
            module_path, param_name = fq_name.rsplit(".", 1) if "." in fq_name else ("", fq_name)
            module = model.get_submodule(module_path) if module_path else model
            parameters.append(getattr(module, param_name))
        return parameters

    def _zero_candidate_grads(self, node_labels: Iterable[str]) -> None:
        for parameter in self._candidate_parameters(node_labels):
            if parameter.grad is not None:
                parameter.grad.zero_()

    def freeze_head(self, candidate: SplitCandidate | str | None = None) -> None:
        chosen = self._candidate_or_default(candidate)
        edge_names = set(self._candidate_param_names(chosen.edge_nodes))
        for name, parameter in self.model.named_parameters():
            if name in edge_names:
                parameter.requires_grad_(False)

    def unfreeze_tail(self, candidate: SplitCandidate | str | None = None) -> None:
        chosen = self._candidate_or_default(candidate)
        cloud_names = set(self._candidate_param_names(chosen.cloud_nodes))
        for name, parameter in self.model.named_parameters():
            if name in cloud_names:
                parameter.requires_grad_(True)

    def get_tail_trainable_params(self, candidate: SplitCandidate | str | None = None) -> list[torch.nn.Parameter]:
        chosen = self._candidate_or_default(candidate)
        return [param for param in self._candidate_parameters(chosen.cloud_nodes) if param.requires_grad]

    def cloud_train_step(
        self,
        payload: SplitPayload | Mapping[str, torch.Tensor] | torch.Tensor,
        targets: Any = None,
        loss_fn=None,
        optimizer: torch.optim.Optimizer | None = None,
        *args: Any,
        candidate: SplitCandidate | str | None = None,
        **kwargs: Any,
    ) -> tuple[Any, torch.Tensor]:
        chosen = self._candidate_or_default(candidate)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        outputs = self.cloud_forward(payload, *args, candidate=chosen, **kwargs)
        loss = loss_fn(outputs, targets) if loss_fn is not None else reduce_output_to_loss(outputs, targets)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        return outputs, loss

    def get_tail_state_dict(self, candidate: SplitCandidate | str | None = None) -> OrderedDict[str, torch.Tensor]:
        chosen = self._candidate_or_default(candidate)
        cloud_names = set(self._candidate_param_names(chosen.cloud_nodes))
        state = OrderedDict()
        for name, tensor in self.model.state_dict().items():
            if any(name == param_name or name.startswith(f"{param_name}.") for param_name in cloud_names):
                state[name] = tensor.detach().cpu().clone()
        return state

    def load_tail_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state_dict, strict=False)
