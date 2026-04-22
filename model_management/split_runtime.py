from __future__ import annotations

import gc
import inspect
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping
import os
import torch
from loguru import logger

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


def _infer_trace_batch_size(graph: GraphIR) -> int:
    try:
        for sample in tuple(graph.sample_args) + tuple(graph.sample_kwargs.values()):
            for tensor in _flatten_tensors(sample):
                if tensor.ndim > 0:
                    return max(1, int(tensor.shape[0]))
    except Exception:
        return 1
    return 1


def _prod_ints(values: list[int]) -> int | None:
    prod = 1
    for value in values:
        if value == -1:
            return None
        if value < 0:
            return None
        prod *= int(value)
    return prod


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


_RUNTIME_DEVICE_FACTORY_FUNCS = {
    "arange",
    "as_tensor",
    "empty",
    "eye",
    "full",
    "linspace",
    "logspace",
    "ones",
    "rand",
    "randint",
    "randn",
    "scalar_tensor",
    "tensor",
    "zeros",
}


def _maybe_inject_runtime_device(
    *,
    func_name: str,
    parent_labels: Iterable[str],
    kwargs: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    resolved_kwargs = dict(kwargs)
    if resolved_kwargs.get("device") is not None:
        return resolved_kwargs
    if list(parent_labels):
        return resolved_kwargs
    if str(func_name).lower() not in _RUNTIME_DEVICE_FACTORY_FUNCS:
        return resolved_kwargs
    resolved_kwargs["device"] = device
    return resolved_kwargs


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


def _call_loss_fn(
    loss_fn,
    outputs: Any,
    targets: Any = None,
    *,
    runtime: "GraphSplitRuntime | None" = None,
    candidate: SplitCandidate | None = None,
) -> torch.Tensor:
    if loss_fn is None:
        return reduce_output_to_loss(outputs, targets)

    kwargs: dict[str, Any] = {}
    try:
        parameters = inspect.signature(loss_fn).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if runtime is not None and ("runtime" in parameters or accepts_kwargs):
        kwargs["runtime"] = runtime
    if candidate is not None and ("candidate" in parameters or accepts_kwargs):
        kwargs["candidate"] = candidate
    return loss_fn(outputs, targets, **kwargs)


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


def _coerce_small_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return int(value.item())
    return None


def _maybe_call_safe_topk(args: list[Any], kwargs: dict[str, Any]) -> Any | None:
    if len(args) < 2:
        return None
    source = args[0]
    if not isinstance(source, torch.Tensor) or source.ndim == 0:
        return None

    k = _coerce_small_int(args[1])
    if k is None:
        return None

    if "dim" in kwargs:
        dim = _coerce_small_int(kwargs["dim"])
    elif len(args) >= 3:
        dim = _coerce_small_int(args[2])
    else:
        dim = -1
    if dim is None:
        return None

    norm_dim = dim if dim >= 0 else source.ndim + dim
    if norm_dim < 0 or norm_dim >= source.ndim:
        return None

    if "largest" in kwargs:
        largest = bool(kwargs["largest"])
    elif len(args) >= 4:
        largest = bool(args[3])
    else:
        largest = True

    if "sorted" in kwargs:
        sorted_flag = bool(kwargs["sorted"])
    elif len(args) >= 5:
        sorted_flag = bool(args[4])
    else:
        sorted_flag = True

    safe_k = max(0, min(int(k), int(source.shape[norm_dim])))
    return source.topk(safe_k, dim=dim, largest=largest, sorted=sorted_flag)


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
        self.trainability_loss_fn = None
        self._validation_cache: dict[tuple[str, float, float], dict[str, Any]] = {}
        self.trace_timings: dict[str, float] = {}
        self.trace_used_output_fallback: bool = False

    def _invalidate_validation_cache(self) -> None:
        self._validation_cache.clear()

    def trace(
        self,
        model: torch.nn.Module,
        sample_input: Any,
        sample_kwargs: Mapping[str, Any] | None = None,
    ) -> "GraphSplitRuntime":
        self.model = model
        grad_state = {
            name: parameter.requires_grad
            for name, parameter in model.named_parameters()
        }
        trainable_param_names = set(grad_state)
        trace_result = trace_model(
            model,
            sample_input,
            sample_kwargs=sample_kwargs,
        )
        if isinstance(trace_result, tuple):
            history, sample_args, sample_kwargs_dict, sample_output = trace_result
            sample_output_spec = None
            output_leaves = None
            self.trace_timings = {}
            self.trace_used_output_fallback = False
        else:
            history = trace_result.history
            sample_args = trace_result.sample_args
            sample_kwargs_dict = trace_result.sample_kwargs
            sample_output = trace_result.sample_output
            sample_output_spec = trace_result.sample_output_spec
            output_leaves = trace_result.output_leaves
            self.trace_timings = dict(trace_result.timings)
            self.trace_used_output_fallback = bool(
                getattr(trace_result, "used_output_fallback", False)
            )
        try:
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(grad_state.get(name, parameter.requires_grad))
            graph_build_started = time.perf_counter()
            self.graph = build_graph_from_trace(
                model,
                history,
                sample_args,
                sample_kwargs_dict,
                sample_output,
                trainable_param_names=trainable_param_names,
                sample_output_spec=sample_output_spec,
                output_leaves=output_leaves,
            )
            self.trace_timings["graph_build"] = time.perf_counter() - graph_build_started
        finally:
            # The runtime only needs the derived graph; retaining the raw
            # TorchLens history keeps large trace-time intermediates alive.
            self.history = None
            cleanup = getattr(history, "cleanup", None)
            if callable(cleanup):
                cleanup()
            del history
            gc.collect()
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(grad_state.get(name, parameter.requires_grad))
        candidate_started = time.perf_counter()
        self.candidates = enumerate_candidates(self.graph)
        self.trace_timings["candidate_enumeration"] = (
            time.perf_counter() - candidate_started
        )
        self.current_candidate = self.candidates[0] if self.candidates else None
        self.reset_runtime_state()
        self._invalidate_validation_cache()
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
        overall_batch_size: int = 1,
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
        kwargs = _maybe_inject_runtime_device(
            func_name=node.func_name,
            parent_labels=node.parent_labels,
            kwargs=kwargs,
            device=self.device,
        )
        args_list = list(args) if isinstance(args, tuple) else list(args)
        kwargs_dict = dict(kwargs)

        trace_batch_size = _infer_trace_batch_size(graph)

        # ==== 动态泛化 Batch Size (仅在 replay batch != trace batch 时启用) ====
        if overall_batch_size > 1 and trace_batch_size != overall_batch_size:
            fn_name = node.func_name.lower()
            
            # --- 1. 处理 Tensor 的形变操作 (第一个参数是 self tensor) ---
            if fn_name in ("view", "reshape") and len(args_list) > 1:
                input_tensor = args_list[0]
                if hasattr(input_tensor, "shape") and input_tensor.ndim > 0:
                    candidate_batch_size = overall_batch_size
                    orig_sh = args_list[1]
                    patched = False
                    
                    # 参数形式 1: args 散装传入 -> tensor.view(4, C, H, W)
                    if len(args_list) > 2 and isinstance(args_list[1], int):
                        if all(isinstance(value, int) for value in args_list[1:]):
                            desired_shape = [int(value) for value in args_list[1:]]
                            desired_prod = _prod_ints(desired_shape)
                            if desired_prod is not None:
                                numel = int(input_tensor.numel())
                                if desired_prod > 0 and desired_prod != numel and numel % desired_prod == 0:
                                    factor = numel // desired_prod
                                    if factor > 1:
                                        desired_shape[0] = int(desired_shape[0]) * int(factor)
                                        args_list[1:] = desired_shape
                                        patched = True
                        if not patched:
                            desired = int(args_list[1])
                            if (
                                int(getattr(input_tensor, "shape")[0]) == candidate_batch_size
                                and desired in (1, trace_batch_size)
                            ):
                                args_list[1] = candidate_batch_size
                                patched = True
                    # 参数形式 2: shape 是 tuple -> tensor.view((4, C, H, W))
                    elif len(args_list) == 2 and isinstance(args_list[1], (tuple, list, torch.Size)):
                        c_shape = list(args_list[1])
                        if c_shape and all(isinstance(item, int) for item in c_shape):
                            desired_shape = [int(item) for item in c_shape]
                            desired_prod = _prod_ints(desired_shape)
                            if desired_prod is not None:
                                numel = int(input_tensor.numel())
                                if desired_prod > 0 and desired_prod != numel and numel % desired_prod == 0:
                                    factor = numel // desired_prod
                                    if factor > 1:
                                        desired_shape[0] = int(desired_shape[0]) * int(factor)
                                        args_list[1] = tuple(desired_shape)
                                        patched = True
                        if not patched and (
                            len(c_shape) > 0
                            and isinstance(c_shape[0], int)
                            and int(getattr(input_tensor, "shape")[0]) == candidate_batch_size
                            and int(c_shape[0]) in (1, trace_batch_size)
                        ):
                            c_shape[0] = candidate_batch_size
                            args_list[1] = tuple(c_shape)
                            patched = True

                    if patched and os.environ.get("DEBUG_SPLIT_RUNTIME") == "1":
                        logger.info("[SplitRuntime] shape patch {} @ {}: {} -> {}", fn_name, label, orig_sh, args_list[1])

            elif fn_name in ("repeat", "tile") and len(args_list) > 1:
                input_tensor = args_list[0]
                if hasattr(input_tensor, "shape") and input_tensor.ndim > 0:
                    candidate_batch_size = overall_batch_size
                    
                    # For repeat/tile, the arguments are repetition factors.
                    # If the trace repeated `trace_batch_size` times, it was likely because the input had batch dim = 1.
                    # If the current input ALREADY has batch dim = candidate_batch_size, we should repeat 1 time, not candidate_batch_size times!
                    # If it still has batch dim = 1, we should repeat candidate_batch_size times.
                    
                    if len(args_list) > 2 and isinstance(args_list[1], int):
                        factor = int(args_list[1])
                        if factor == trace_batch_size:
                            in_batch = int(getattr(input_tensor, "shape")[0])
                            args_list[1] = 1 if in_batch == candidate_batch_size else candidate_batch_size
                    
                    elif len(args_list) == 2 and isinstance(args_list[1], (tuple, list, torch.Size)):
                        c_shape = list(args_list[1])
                        if len(c_shape) > 0 and isinstance(c_shape[0], int) and int(c_shape[0]) == trace_batch_size:
                            in_batch = int(getattr(input_tensor, "shape")[0])
                            c_shape[0] = 1 if in_batch == candidate_batch_size else candidate_batch_size
                            args_list[1] = tuple(c_shape)
                            
            # --- 2. 处理工厂函数 (无 self tensor, 参数即 shape) ---
            elif fn_name == "expand" and len(args_list) > 1:
                input_tensor = args_list[0]
                if hasattr(input_tensor, "shape") and input_tensor.ndim > 0:
                    candidate_batch_size = overall_batch_size
                    orig_sh = args_list[1]
                    patched = False

                    # tensor.expand(B, ...)
                    if len(args_list) > 2 and isinstance(args_list[1], int):
                        desired = int(args_list[1])
                        if desired in (1, trace_batch_size):
                            args_list[1] = candidate_batch_size
                            patched = True

                    # tensor.expand((B, ...))
                    elif len(args_list) == 2 and isinstance(args_list[1], (tuple, list, torch.Size)):
                        c_shape = list(args_list[1])
                        if len(c_shape) > 0 and isinstance(c_shape[0], int) and int(c_shape[0]) in (1, trace_batch_size):
                            c_shape[0] = candidate_batch_size
                            args_list[1] = tuple(c_shape)
                            patched = True
            elif fn_name in ("zeros", "ones", "empty", "randn", "full"):
                candidate_batch_size = overall_batch_size
                orig_sh = None
                patched = False
                # 检查 kwargs 里是否有 size
                if "size" in kwargs_dict and isinstance(kwargs_dict["size"], (list, tuple, torch.Size)):
                    orig_sh = kwargs_dict["size"]
                    c_shape = list(kwargs_dict["size"])
                    if len(c_shape) > 0 and isinstance(c_shape[0], int) and int(c_shape[0]) in (1, trace_batch_size):
                        c_shape[0] = candidate_batch_size
                        kwargs_dict["size"] = tuple(c_shape)
                        patched = True
                else:
                    # tuple 形式如 zeros((4, C, H), ...)
                    if len(args_list) > 0 and isinstance(args_list[0], (tuple, list, torch.Size)):
                        orig_sh = args_list[0]
                        c_shape = list(args_list[0])
                        if len(c_shape) > 0 and isinstance(c_shape[0], int) and int(c_shape[0]) in (1, trace_batch_size):
                            c_shape[0] = candidate_batch_size
                            args_list[0] = tuple(c_shape)
                            patched = True
                    # 散装形式如 zeros(4, C, H, ...)
                    elif len(args_list) > 0 and isinstance(args_list[0], int):
                        orig_sh = args_list[0]
                        desired = int(args_list[0])
                        if desired in (1, trace_batch_size):
                            args_list[0] = candidate_batch_size
                            patched = True
                            
        args = tuple(args_list)
        # =========================================================================================

        try:
            safe_topk = None
            if node.func_name.lower() == "topk":
                safe_topk = _maybe_call_safe_topk(args_list, kwargs_dict)
            if node.func_name.lower() == "cat" and overall_batch_size > 1 and trace_batch_size != overall_batch_size:
                dim = kwargs_dict.get("dim", 0)
                pieces = args_list[0] if args_list and isinstance(args_list[0], (list, tuple)) else None
                if pieces and all(hasattr(piece, "ndim") for piece in pieces):
                    tensors = list(pieces)
                    try:
                        ref_ndim = int(tensors[0].ndim)
                    except AttributeError:
                        ref_ndim = -1
                    if ref_ndim > 0 and all(int(getattr(t, "ndim", -1)) == ref_ndim for t in tensors):
                        if isinstance(dim, int) and dim < 0:
                            dim = ref_ndim + int(dim)
                        desired_sizes = [1] * ref_ndim
                        for axis in range(ref_ndim):
                            if axis == int(dim):
                                continue
                            desired_sizes[axis] = max(int(t.shape[axis]) for t in tensors)
                        patched = False
                        patched_tensors: list[torch.Tensor] = []
                        for t in tensors:
                            shape = list(t.shape)
                            expand_shape: list[int] = [-1] * ref_ndim
                            can_expand = False
                            for axis in range(ref_ndim):
                                if axis == int(dim):
                                    continue
                                want = int(desired_sizes[axis])
                                have = int(shape[axis])
                                if want == have:
                                    continue
                                if have == 1 and want > 1:
                                    expand_shape[axis] = want
                                    can_expand = True
                                else:
                                    can_expand = False
                                    break
                            if can_expand:
                                patched_tensors.append(t.expand(*expand_shape))
                                patched = True
                            else:
                                patched_tensors.append(t)
                        if patched:
                            args_list[0] = tuple(patched_tensors) if isinstance(pieces, tuple) else patched_tensors
            output = safe_topk if safe_topk is not None else node.func(*args, **kwargs_dict)
        except RuntimeError:
            if node.func_name.lower() == "cat":
                dim = kwargs_dict.get("dim", 0)
                pieces = args_list[0] if args_list and isinstance(args_list[0], (list, tuple)) else ()
                shapes = [tuple(t.shape) for t in pieces if isinstance(t, torch.Tensor)]
                mismatch_axes: list[int] = []
                if shapes:
                    ref = shapes[0]
                    ref_ndim = len(ref)
                    if isinstance(dim, int) and dim < 0:
                        dim = ref_ndim + int(dim)
                    for axis in range(ref_ndim):
                        if axis == int(dim):
                            continue
                        sizes = {shape[axis] for shape in shapes if len(shape) == ref_ndim}
                        if len(sizes) > 1:
                            mismatch_axes.append(axis)
                logger.error(
                    "[SplitRuntime] cat failed @ {} (dim={}) overall_batch_size={} trace_batch_size={} mismatch_axes={} shapes={} parent_labels={}",
                    label,
                    dim,
                    overall_batch_size,
                    locals().get("trace_batch_size", None),
                    mismatch_axes,
                    shapes,
                    list(getattr(node, "parent_labels", ())),
                )
                if True:
                    parents = {}
                    for parent_label in getattr(node, "parent_labels", ()):
                        value = available_tensors.get(parent_label)
                        if isinstance(value, torch.Tensor):
                            parents[parent_label] = {"type": "tensor", "shape": tuple(value.shape), "dtype": str(value.dtype)}
                        else:
                            parents[parent_label] = {"type": type(value).__name__}
                    logger.info("[SplitRuntime] cat parent snapshots @ {} -> {}", label, parents)
            raise
        except IndexError:
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
        
        overall_batch_size = 1
        for val in initial_tensors.values():
            if isinstance(val, torch.Tensor) and val.ndim > 0:
                overall_batch_size = val.shape[0]
                break

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
                overall_batch_size=overall_batch_size,
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
        cache_key = (chosen.candidate_id, float(atol), float(rtol))
        cached = self._validation_cache.get(cache_key)
        if cached is not None:
            chosen.is_validated = bool(cached.get("success", False))
            chosen.is_trainable_tail = bool(cached.get("tail_trainability", chosen.is_trainable_tail))
            chosen.validation_error = cached.get("error")
            return dict(cached)
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
        if success:
            trainability = self._check_tail_trainability(chosen, payload.detach(), *sample_args, **sample_kwargs)

        chosen.is_validated = success
        chosen.is_trainable_tail = trainability
        chosen.validation_error = None if success else f"max_diff={max_diff}"
        report = {
            "success": success,
            "max_diff": max_diff,
            "edge_latency": edge_elapsed,
            "cloud_latency": cloud_elapsed,
            "end_to_end_latency": edge_elapsed + cloud_elapsed,
            "tail_trainability": trainability,
            "stability_score": 1.0 if success else 0.0,
            "error": chosen.validation_error,
        }
        self._validation_cache[cache_key] = dict(report)
        return report

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
        grad_state = {
            name: parameter.requires_grad
            for name, parameter in self.model.named_parameters()
        }
        self.freeze_head(candidate)
        self.unfreeze_tail(candidate)
        self._zero_candidate_grads(candidate.cloud_nodes)
        outputs = self.cloud_forward(payload, *args, candidate=candidate, **kwargs)
        if self.trainability_loss_fn is not None:
            loss = _call_loss_fn(
                self.trainability_loss_fn,
                outputs,
                _make_dummy_detection_targets(args, device=self.device),
                runtime=self,
                candidate=candidate,
            )
        else:
            loss = reduce_output_to_loss(outputs)
        try:
            loss.backward()
        except Exception:
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad_(grad_state.get(name, parameter.requires_grad))
            self._zero_candidate_grads(candidate.cloud_nodes)
            return False
        has_grad = any(param.grad is not None and torch.count_nonzero(param.grad).item() > 0 for param in params)
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(grad_state.get(name, parameter.requires_grad))
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
        effective_loss_fn = loss_fn if loss_fn is not None else self.trainability_loss_fn
        loss = _call_loss_fn(
            effective_loss_fn,
            outputs,
            targets,
            runtime=self,
            candidate=chosen,
        )
        if optimizer is not None:
            loss.backward()
            optimizer.step()
            self._invalidate_validation_cache()
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
        self._invalidate_validation_cache()


def _make_dummy_detection_targets(
    args: tuple[Any, ...],
    *,
    device: torch.device,
) -> dict[str, Any]:
    input_height, input_width = _infer_input_image_size_from_args(args)
    return {
        "boxes": [[0.25 * input_width, 0.25 * input_height, 0.75 * input_width, 0.75 * input_height]],
        "labels": [1],
        "_split_meta": {
            "input_image_size": [input_height, input_width],
        },
    }


def _infer_input_image_size_from_args(args: tuple[Any, ...]) -> tuple[int, int]:
    for arg in args:
        tensor = _extract_first_input_tensor(arg)
        if tensor is not None and tensor.ndim >= 3:
            return int(tensor.shape[-2]), int(tensor.shape[-1])
    return 224, 224


def _extract_first_input_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, list):
        for item in value:
            tensor = _extract_first_input_tensor(item)
            if tensor is not None:
                return tensor
        return None
    if isinstance(value, tuple):
        for item in value:
            tensor = _extract_first_input_tensor(item)
            if tensor is not None:
                return tensor
        return None
    if isinstance(value, dict):
        for item in value.values():
            tensor = _extract_first_input_tensor(item)
            if tensor is not None:
                return tensor
    return None
