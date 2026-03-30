from __future__ import annotations

import copy
import dataclasses
import inspect
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import torch

try:
    import torchlens as tl
except ImportError:  # pragma: no cover - guarded by caller
    tl = None


SKIPPED_FUNCTION_NAMES = {"isinf", "isnan", "nantonum", "nan_to_num"}


def _get_attr(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _normalize_path_key(path: Any) -> tuple[Any, ...]:
    if isinstance(path, tuple):
        return path
    return (path,)


def _safe_deepcopy(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        values = {
            field.name: _safe_deepcopy(getattr(obj, field.name))
            for field in dataclasses.fields(obj)
        }
        try:
            return type(obj)(**values)
        except Exception:
            return values
    if isinstance(obj, list):
        return [_safe_deepcopy(item) for item in obj]
    if isinstance(obj, tuple) and hasattr(type(obj), "_fields"):
        values = [_safe_deepcopy(item) for item in obj]
        try:
            return type(obj)(*values)
        except Exception:
            return tuple(values)
    if isinstance(obj, tuple):
        return tuple(_safe_deepcopy(item) for item in obj)
    if isinstance(obj, dict):
        values = {key: _safe_deepcopy(value) for key, value in obj.items()}
        if type(obj) is dict:
            return values
        try:
            return type(obj)(values)
        except Exception:
            return values
    try:
        return copy.deepcopy(obj)
    except Exception:
        return obj


def _resolve_attr_path(root: Any, dotted_path: str | None) -> Any:
    if not dotted_path:
        return root
    current = root
    for part in dotted_path.split("."):
        if isinstance(current, (list, tuple)) and part.isdigit():
            current = current[int(part)]
        elif isinstance(current, dict):
            current = current[part]
        else:
            current = getattr(current, part)
    return current


def _index_nested(obj: Any, path: Sequence[Any]) -> Any:
    current = obj
    for part in path:
        if dataclasses.is_dataclass(current) and not isinstance(current, type):
            current = getattr(current, str(part))
        else:
            current = current[part]
    return current


def _clone_tree_tensors(tree: Any, *, device: torch.device | None = None, detach: bool = False) -> Any:
    memo: dict[int, Any] = {}

    def _clone(value: Any) -> Any:
        value_id = id(value)
        if value_id in memo:
            return memo[value_id]
        if isinstance(value, torch.Tensor):
            cloned = value.detach().clone() if detach else value.clone()
            if device is not None:
                cloned = cloned.to(device)
            memo[value_id] = cloned
            return cloned
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            kwargs = {
                field.name: _clone(getattr(value, field.name))
                for field in dataclasses.fields(value)
            }
            try:
                cloned = type(value)(**kwargs)
            except Exception:
                cloned = kwargs
            memo[value_id] = cloned
            return cloned
        if isinstance(value, list):
            cloned_list: list[Any] = []
            memo[value_id] = cloned_list
            cloned_list.extend(_clone(item) for item in value)
            return cloned_list
        if isinstance(value, tuple):
            items = [_clone(item) for item in value]
            if hasattr(type(value), "_fields"):
                try:
                    cloned_tuple = type(value)(*items)
                except Exception:
                    cloned_tuple = tuple(items)
            else:
                cloned_tuple = tuple(items)
            memo[value_id] = cloned_tuple
            return cloned_tuple
        if isinstance(value, dict):
            cloned_dict = {key: _clone(item) for key, item in value.items()}
            if type(value) is not dict:
                try:
                    cloned_dict = type(value)(cloned_dict)
                except Exception:
                    pass
            memo[value_id] = cloned_dict
            return cloned_dict
        try:
            cloned = copy.copy(value)
        except Exception:
            cloned = value
        memo[value_id] = cloned
        return cloned

    return _clone(tree)


def _tensor_exact_match(candidate: Any, target: torch.Tensor) -> bool:
    if not isinstance(candidate, torch.Tensor):
        return False
    if candidate.shape != target.shape or candidate.dtype != target.dtype:
        return False
    candidate_compare, target_compare = _prepare_tensors_for_exact_match(candidate, target)
    return torch.equal(candidate_compare, target_compare)


def _tensors_need_device_alignment(candidate: torch.Tensor, target: torch.Tensor) -> bool:
    return candidate.device != target.device


def _prepare_tensors_for_exact_match(
    candidate: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidate_compare = candidate.detach()
    target_compare = target.detach()
    if _tensors_need_device_alignment(candidate_compare, target_compare):
        return candidate_compare.cpu(), target_compare.cpu()
    return candidate_compare, target_compare


def _find_tensor_path(container: Any, target: torch.Tensor) -> tuple[Any, ...] | None:
    if isinstance(container, torch.Tensor):
        return () if _tensor_exact_match(container, target) else None
    if dataclasses.is_dataclass(container) and not isinstance(container, type):
        for field in dataclasses.fields(container):
            subpath = _find_tensor_path(getattr(container, field.name), target)
            if subpath is not None:
                return (field.name,) + subpath
        return None
    if isinstance(container, (list, tuple)):
        for index, value in enumerate(container):
            subpath = _find_tensor_path(value, target)
            if subpath is not None:
                return (index,) + subpath
        return None
    if isinstance(container, dict):
        for key, value in container.items():
            subpath = _find_tensor_path(value, target)
            if subpath is not None:
                return (key,) + subpath
        return None
    return None


def _match_leaf_addresses_to_labels(
    labels: Sequence[str],
    leaves: Mapping[str, torch.Tensor],
    layer_lookup: Mapping[str, Any],
) -> dict[str, str]:
    """Infer leaf-address mappings by matching sampled leaf tensors to layer activations."""
    matched: dict[str, str] = {}
    used_addresses: set[str] = set()

    for label in labels:
        layer = layer_lookup.get(label)
        activation = _get_attr(layer, "activation", default=None)
        if not isinstance(activation, torch.Tensor):
            continue
        for address, tensor in leaves.items():
            if address in used_addresses:
                continue
            if _tensor_exact_match(tensor, activation):
                matched[address] = label
                used_addresses.add(address)
                break
    return matched


def _infer_parent_output_path(parent_layer: Any, child_label: str) -> tuple[Any, ...] | None:
    child_versions = _get_attr(parent_layer, "children_tensor_versions", default={}) or {}
    if child_label not in child_versions:
        return None
    child_saved_value = child_versions[child_label]
    parent_activation = _get_attr(parent_layer, "activation", default=None)
    if not isinstance(child_saved_value, torch.Tensor) or parent_activation is None:
        return None
    return _find_tensor_path(parent_activation, child_saved_value)


def _infer_child_specific_frozen_value(
    *,
    parent_layer: Any,
    child_label: str,
    captured_container: Any,
    location: tuple[Any, ...],
) -> Any:
    child_versions = _get_attr(parent_layer, "children_tensor_versions", default={}) or {}
    child_saved_value = child_versions.get(child_label)
    if not isinstance(child_saved_value, torch.Tensor):
        return None
    if child_saved_value.requires_grad:
        return None
    if _infer_parent_output_path(parent_layer, child_label) is not None:
        return None
    try:
        captured_value = _index_nested(captured_container, location)
    except Exception:
        return None
    if not _tensor_exact_match(child_saved_value, captured_value):
        return None
    return _clone_tree_tensors(child_saved_value.detach().cpu(), detach=True)


def _infer_func_name(layer: Any) -> str:
    name = _get_attr(layer, "func_applied_name", "func_name", default=None)
    if isinstance(name, str) and name:
        return name
    func = _get_attr(layer, "func_applied", default=None)
    inferred = getattr(func, "__name__", None)
    if isinstance(inferred, str) and inferred:
        return inferred
    layer_type = _get_attr(layer, "layer_type", default=None)
    if isinstance(layer_type, str) and layer_type:
        return layer_type
    return "none"


def _layer_creation_args(layer: Any) -> list[Any]:
    args = _get_attr(layer, "creation_args", default=None)
    if args is None:
        args = _get_attr(layer, "captured_args", default=None)
    return list(_safe_deepcopy(args) or [])


def _layer_creation_kwargs(layer: Any) -> dict[str, Any]:
    kwargs = _get_attr(layer, "creation_kwargs", default=None)
    if kwargs is None:
        kwargs = _get_attr(layer, "captured_kwargs", default=None)
    return dict(_safe_deepcopy(kwargs) or {})


@dataclass(frozen=True)
class ParentTensorRef:
    parent_label: str
    path: tuple[Any, ...] | None = None
    frozen_value: Any = None


@dataclass(frozen=True)
class ParameterTensorRef:
    module_path: str
    param_name: str
    fq_name: str


@dataclass(frozen=True)
class BufferTensorRef:
    module_path: str
    buffer_name: str
    fq_name: str


@dataclass(frozen=True)
class ConstantTensorRef:
    tensor: torch.Tensor


ArgPlaceholder = ParentTensorRef | ParameterTensorRef | BufferTensorRef | ConstantTensorRef


@dataclass
class TreeSpec:
    kind: str
    address: str | None = None
    value: Any = None
    children: Any = None
    type_ref: Any = None


def build_tree_spec(obj: Any, prefix: str) -> tuple[TreeSpec, OrderedDict[str, torch.Tensor]]:
    leaves: "OrderedDict[str, torch.Tensor]" = OrderedDict()

    def _build(current: Any, address: str) -> TreeSpec:
        if isinstance(current, torch.Tensor):
            leaves[address] = current
            return TreeSpec(kind="tensor", address=address)
        if dataclasses.is_dataclass(current):
            fields = OrderedDict()
            for field_info in dataclasses.fields(current):
                fields[field_info.name] = _build(getattr(current, field_info.name), f"{address}.{field_info.name}")
            return TreeSpec(kind="dataclass", type_ref=type(current), children=fields)
        if isinstance(current, tuple) and hasattr(current, "_fields"):
            fields = OrderedDict()
            for field_name in current._fields:
                fields[field_name] = _build(getattr(current, field_name), f"{address}.{field_name}")
            return TreeSpec(kind="namedtuple", type_ref=type(current), children=fields)
        if isinstance(current, list):
            return TreeSpec(
                kind="list",
                children=[_build(item, f"{address}.{index}") for index, item in enumerate(current)],
            )
        if isinstance(current, tuple):
            return TreeSpec(
                kind="tuple",
                type_ref=type(current),
                children=[_build(item, f"{address}.{index}") for index, item in enumerate(current)],
            )
        if isinstance(current, dict):
            fields = OrderedDict()
            for key, value in current.items():
                fields[key] = _build(value, f"{address}.{key}")
            return TreeSpec(kind="dict", type_ref=type(current), children=fields)
        return TreeSpec(kind="const", value=_safe_deepcopy(current))

    return _build(obj, prefix), leaves


def materialize_tree_spec(spec: TreeSpec, tensor_map: Mapping[str, torch.Tensor]) -> Any:
    if spec.kind == "tensor":
        return tensor_map[spec.address]
    if spec.kind == "const":
        return copy.deepcopy(spec.value)
    if spec.kind == "list":
        return [materialize_tree_spec(child, tensor_map) for child in spec.children]
    if spec.kind == "tuple":
        items = [materialize_tree_spec(child, tensor_map) for child in spec.children]
        tuple_type = spec.type_ref or tuple
        if tuple_type is tuple:
            return tuple(items)
        try:
            return tuple_type(items)
        except Exception:
            return tuple(items)
    if spec.kind == "dict":
        rebuilt = {key: materialize_tree_spec(child, tensor_map) for key, child in spec.children.items()}
        dict_type = spec.type_ref or dict
        if dict_type is dict:
            return rebuilt
        try:
            return dict_type(rebuilt)
        except Exception:
            return rebuilt
    if spec.kind == "namedtuple":
        values = {key: materialize_tree_spec(child, tensor_map) for key, child in spec.children.items()}
        tuple_type = spec.type_ref
        if tuple_type is None:
            return values
        try:
            return tuple_type(**values)
        except Exception:
            try:
                return tuple_type(*values.values())
            except Exception:
                return values
    if spec.kind == "dataclass":
        values = {key: materialize_tree_spec(child, tensor_map) for key, child in spec.children.items()}
        dataclass_type = spec.type_ref
        if dataclass_type is None:
            return values
        try:
            return dataclass_type(**values)
        except Exception:
            return values
    raise ValueError(f"Unsupported TreeSpec kind: {spec.kind}")


def _replace_placeholders(obj: Any, resolver) -> Any:
    if isinstance(obj, (ParentTensorRef, ParameterTensorRef, BufferTensorRef, ConstantTensorRef)):
        return resolver(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        values = {
            field.name: _replace_placeholders(getattr(obj, field.name), resolver)
            for field in dataclasses.fields(obj)
        }
        try:
            return type(obj)(**values)
        except Exception:
            return values
    if isinstance(obj, list):
        return [_replace_placeholders(item, resolver) for item in obj]
    if isinstance(obj, tuple):
        values = [_replace_placeholders(item, resolver) for item in obj]
        if hasattr(type(obj), "_fields"):
            try:
                return type(obj)(*values)
            except Exception:
                return tuple(values)
        return tuple(values)
    if isinstance(obj, dict):
        rebuilt = {key: _replace_placeholders(value, resolver) for key, value in obj.items()}
        if type(obj) is dict:
            return rebuilt
        try:
            return type(obj)(rebuilt)
        except Exception:
            return rebuilt
    return _safe_deepcopy(obj)


def normalize_runtime_inputs(
    signature: inspect.Signature,
    *args: Any,
    partial: bool = False,
    **kwargs: Any,
) -> inspect.BoundArguments:
    if partial:
        return signature.bind_partial(*args, **kwargs)
    return signature.bind(*args, **kwargs)


def flatten_bound_inputs(bound: inspect.BoundArguments) -> "OrderedDict[str, torch.Tensor]":
    flattened: "OrderedDict[str, torch.Tensor]" = OrderedDict()

    def _walk(value: Any, prefix: str) -> None:
        if isinstance(value, torch.Tensor):
            flattened[prefix] = value
            return
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field in dataclasses.fields(value):
                _walk(getattr(value, field.name), f"{prefix}.{field.name}")
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                _walk(item, f"{prefix}.{index}")
            return
        if isinstance(value, tuple):
            if hasattr(type(value), "_fields"):
                for field in value._fields:
                    _walk(getattr(value, field), f"{prefix}.{field}")
                return
            for index, item in enumerate(value):
                _walk(item, f"{prefix}.{index}")
            return
        if isinstance(value, dict):
            for key, item in value.items():
                _walk(item, f"{prefix}.{key}")

    for name, value in bound.arguments.items():
        _walk(value, f"input.{name}")
    return flattened


@dataclass
class GraphNode:
    label: str
    node_type: str
    func_name: str
    func: Any
    parent_labels: list[str]
    child_labels: list[str]
    parent_arg_locations: dict[str, dict[tuple[Any, ...], str]]
    arg_template: list[Any]
    kwarg_template: dict[str, Any]
    non_tensor_args: dict[str, Any]
    parameter_refs: list[ParameterTensorRef]
    buffer_refs: list[BufferTensorRef]
    input_output_address: str | None
    containing_module: str | None
    containing_modules: list[str]
    is_input: bool
    is_output: bool
    is_inplace: bool
    is_multi_output: bool
    multi_output_index: int | None
    aggregation_kind: str | None
    indexing_metadata: Any
    tensor_shape: tuple[int, ...] | None
    tensor_dtype: torch.dtype | None
    numel: int
    estimated_bytes: int
    estimated_flops: float
    has_trainable_params: bool
    topological_index: int = 0
    depth_from_input: int = 0
    depth_from_output: int = 0

    def resolve_args(
        self,
        available_tensors: Mapping[str, Any],
        model: torch.nn.Module,
        device: torch.device | None = None,
        clone_parent_labels: set[str] | None = None,
        clone_cache: dict[tuple[str, tuple[Any, ...] | None], Any] | None = None,
    ) -> tuple[list[Any], dict[str, Any]]:
        clone_parent_labels = clone_parent_labels or set()
        clone_cache = clone_cache or {}

        def _resolve(ref: ArgPlaceholder) -> Any:
            if isinstance(ref, ParentTensorRef):
                if ref.frozen_value is not None:
                    return _clone_tree_tensors(ref.frozen_value, device=device, detach=True)
                value = available_tensors[ref.parent_label]
                if ref.path:
                    value = _index_nested(value, ref.path)
                if ref.parent_label not in clone_parent_labels:
                    return value
                cache_key = (ref.parent_label, ref.path)
                if cache_key not in clone_cache:
                    clone_cache[cache_key] = _clone_tree_tensors(value, device=device, detach=False)
                return clone_cache[cache_key]
            if isinstance(ref, ParameterTensorRef):
                module = _resolve_attr_path(model, ref.module_path)
                return getattr(module, ref.param_name)
            if isinstance(ref, BufferTensorRef):
                module = _resolve_attr_path(model, ref.module_path)
                try:
                    return module.get_buffer(ref.buffer_name)
                except (AttributeError, TypeError):
                    if isinstance(module, (list, tuple)) and ref.buffer_name.isdigit():
                        return module[int(ref.buffer_name)]
                    if isinstance(module, dict):
                        return module[ref.buffer_name]
                    return getattr(module, ref.buffer_name)
            if isinstance(ref, ConstantTensorRef):
                tensor = ref.tensor
                if device is None:
                    return tensor
                return tensor.to(device=device)
            raise TypeError(f"Unsupported placeholder type: {type(ref)!r}")

        return (
            _replace_placeholders(self.arg_template, _resolve),
            _replace_placeholders(self.kwarg_template, _resolve),
        )


@dataclass
class GraphIR:
    nodes: "OrderedDict[str, GraphNode]"
    input_labels: list[str]
    output_labels: list[str]
    topological_order: list[str]
    relevant_labels: list[str]
    input_spec: TreeSpec
    output_spec: TreeSpec
    input_address_to_label: dict[str, str]
    output_address_to_label: dict[str, str]
    forward_signature: inspect.Signature
    sample_args: tuple[Any, ...]
    sample_kwargs: dict[str, Any]
    sample_input_spec: TreeSpec
    sample_output_spec: TreeSpec

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)


def estimate_node_flops(layer_type: str, output_shape: tuple[int, ...] | None, creation_args: Sequence[Any]) -> float:
    if not output_shape:
        return 0.0
    output_elements = float(int(torch.tensor(output_shape).prod().item())) if output_shape else 0.0
    layer_type = (layer_type or "").lower()
    if layer_type in {"conv2d", "conv", "conv1d", "conv3d"} and len(creation_args) >= 2:
        weight = creation_args[1]
        if isinstance(weight, torch.Tensor) and weight.dim() >= 3:
            kernel_mul = float(weight[0].numel())
            return 2.0 * output_elements * kernel_mul
    if layer_type in {"linear", "addmm", "matmul"} and len(creation_args) >= 2:
        weight = creation_args[1]
        if isinstance(weight, torch.Tensor):
            return 2.0 * output_elements * float(weight.shape[-1])
    if layer_type in {"batchnorm", "batch_norm"}:
        return 4.0 * output_elements
    if layer_type in {"relu", "gelu", "sigmoid", "tanh", "softmax", "dropout"}:
        return output_elements
    if layer_type in {"add", "__add__", "mul", "__mul__", "div", "__div__", "sub", "__sub__"}:
        return output_elements
    return output_elements


def _tensor_ref_from_address(address: str | None, *, kind: str) -> ParameterTensorRef | BufferTensorRef | None:
    if not address:
        return None
    if "." in address:
        module_path, name = address.rsplit(".", 1)
    else:
        module_path, name = "", address
    fq_name = f"{module_path}.{name}" if module_path else name
    if kind == "param":
        return ParameterTensorRef(module_path=module_path, param_name=name, fq_name=fq_name)
    return BufferTensorRef(module_path=module_path, buffer_name=name, fq_name=fq_name)


def _iter_constant_tensor_paths(value: Any, path: tuple[Any, ...] = ()) -> Iterator[tuple[tuple[Any, ...], ConstantTensorRef]]:
    if isinstance(value, ConstantTensorRef):
        yield path, value
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from _iter_constant_tensor_paths(getattr(value, field.name), path + (field.name,))
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_constant_tensor_paths(item, path + (index,))
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_constant_tensor_paths(item, path + (key,))


def _apply_placeholder_at_path(template: Any, path: tuple[Any, ...], placeholder: ArgPlaceholder) -> Any:
    if not path:
        return placeholder
    head, *tail = path
    if dataclasses.is_dataclass(template) and not isinstance(template, type):
        values = {field.name: getattr(template, field.name) for field in dataclasses.fields(template)}
        values[str(head)] = _apply_placeholder_at_path(values[str(head)], tuple(tail), placeholder)
        try:
            return type(template)(**values)
        except Exception:
            return values
    if isinstance(template, list):
        updated = list(template)
        updated[int(head)] = _apply_placeholder_at_path(updated[int(head)], tuple(tail), placeholder)
        return updated
    if isinstance(template, tuple):
        updated = list(template)
        updated[int(head)] = _apply_placeholder_at_path(updated[int(head)], tuple(tail), placeholder)
        if hasattr(type(template), "_fields"):
            try:
                return type(template)(*updated)
            except Exception:
                return tuple(updated)
        return tuple(updated)
    if isinstance(template, dict):
        updated = dict(template)
        updated[head] = _apply_placeholder_at_path(updated[head], tuple(tail), placeholder)
        if type(template) is not dict:
            try:
                return type(template)(updated)
            except Exception:
                return updated
        return updated
    return template


def _find_matching_parent_ref(
    target: torch.Tensor,
    *,
    current_label: str,
    previous_layers: Sequence[Any],
) -> ParentTensorRef | None:
    for parent_layer in reversed(list(previous_layers)):
        parent_label = _get_attr(parent_layer, "layer_label", default=None)
        if parent_label is None:
            continue
        child_versions = _get_attr(parent_layer, "children_tensor_versions", default={}) or {}
        child_version = child_versions.get(current_label)
        if _tensor_exact_match(child_version, target):
            path = _find_tensor_path(_get_attr(parent_layer, "activation", default=None), child_version)
            frozen_value = None
            if path is None and isinstance(child_version, torch.Tensor) and not child_version.requires_grad:
                frozen_value = _clone_tree_tensors(child_version.detach().cpu(), detach=True)
            return ParentTensorRef(parent_label=parent_label, path=path, frozen_value=frozen_value)
        parent_activation = _get_attr(parent_layer, "activation", default=None)
        path = _find_tensor_path(parent_activation, target)
        if path is not None:
            return ParentTensorRef(parent_label=parent_label, path=path)
    return None


def _build_arg_templates(
    layer: Any,
    *,
    previous_layers: Sequence[Any],
    parent_layer_lookup: Mapping[str, Any],
) -> tuple[list[Any], dict[str, Any], list[ParameterTensorRef], list[str]]:
    creation_args = _layer_creation_args(layer)
    creation_kwargs = _layer_creation_kwargs(layer)
    current_label = _get_attr(layer, "layer_label", default="")

    parent_arg_locations_raw = _get_attr(layer, "parent_layer_arg_locs", default={}) or {}
    parent_arg_locations = {
        "args": {
            _normalize_path_key(path): label
            for path, label in dict(parent_arg_locations_raw.get("args", {})).items()
        },
        "kwargs": {
            _normalize_path_key(path): label
            for path, label in dict(parent_arg_locations_raw.get("kwargs", {})).items()
        },
    }

    param_refs: list[ParameterTensorRef] = []
    param_queue: list[ParameterTensorRef] = []
    for param_log in list(_get_attr(layer, "parent_param_logs", default=[])) or []:
        module_path = _get_attr(param_log, "module_address", default="") or ""
        param_name = _get_attr(param_log, "name", default="")
        fq_name = f"{module_path}.{param_name}" if module_path else param_name
        ref = ParameterTensorRef(module_path=module_path, param_name=param_name, fq_name=fq_name)
        param_refs.append(ref)
        param_queue.append(ref)

    inferred_parent_labels: list[str] = []

    def _mark(obj: Any, location_kind: str, prefix: tuple[Any, ...] = ()) -> Any:
        if isinstance(obj, torch.Tensor):
            param_address = getattr(obj, "tl_param_address", None)
            if param_address:
                ref = _tensor_ref_from_address(param_address, kind="param")
                if isinstance(ref, ParameterTensorRef):
                    if all(existing.fq_name != ref.fq_name for existing in param_refs):
                        param_refs.append(ref)
                    return ref
            buffer_address = getattr(obj, "tl_buffer_address", None)
            if buffer_address:
                ref = _tensor_ref_from_address(buffer_address, kind="buffer")
                if isinstance(ref, BufferTensorRef):
                    return ref
            parent_label = parent_arg_locations[location_kind].get(prefix)
            if parent_label is not None:
                parent_layer = parent_layer_lookup.get(parent_label)
                inferred_parent_labels.append(parent_label)
                return ParentTensorRef(
                    parent_label=parent_label,
                    path=_infer_parent_output_path(parent_layer, current_label) if parent_layer is not None else None,
                    frozen_value=(
                        _infer_child_specific_frozen_value(
                            parent_layer=parent_layer,
                            child_label=current_label,
                            captured_container=creation_args if location_kind == "args" else creation_kwargs,
                            location=prefix,
                        )
                        if parent_layer is not None
                        else None
                    ),
                )
            if param_queue:
                return param_queue.pop(0)
            return ConstantTensorRef(obj.detach().cpu())
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            values = {
                field.name: _mark(getattr(obj, field.name), location_kind, prefix + (field.name,))
                for field in dataclasses.fields(obj)
            }
            try:
                return type(obj)(**values)
            except Exception:
                return values
        if isinstance(obj, list):
            return [_mark(item, location_kind, prefix + (index,)) for index, item in enumerate(obj)]
        if isinstance(obj, tuple):
            values = [_mark(item, location_kind, prefix + (index,)) for index, item in enumerate(obj)]
            if hasattr(type(obj), "_fields"):
                try:
                    return type(obj)(*values)
                except Exception:
                    return tuple(values)
            return tuple(values)
        if isinstance(obj, dict):
            values = {key: _mark(value, location_kind, prefix + (key,)) for key, value in obj.items()}
            if type(obj) is not dict:
                try:
                    return type(obj)(values)
                except Exception:
                    return values
            return values
        return _safe_deepcopy(obj)

    arg_template = _mark(creation_args, "args", ())
    kwarg_template = _mark(creation_kwargs, "kwargs", ())

    for path, tensor_const in list(_iter_constant_tensor_paths(arg_template)):
        parent_ref = _find_matching_parent_ref(
            tensor_const.tensor,
            current_label=current_label,
            previous_layers=previous_layers,
        )
        if parent_ref is None:
            continue
        arg_template = _apply_placeholder_at_path(arg_template, path, parent_ref)
        inferred_parent_labels.append(parent_ref.parent_label)

    for path, tensor_const in list(_iter_constant_tensor_paths(kwarg_template)):
        parent_ref = _find_matching_parent_ref(
            tensor_const.tensor,
            current_label=current_label,
            previous_layers=previous_layers,
        )
        if parent_ref is None:
            continue
        kwarg_template = _apply_placeholder_at_path(kwarg_template, path, parent_ref)
        inferred_parent_labels.append(parent_ref.parent_label)

    return arg_template, kwarg_template, param_refs, inferred_parent_labels


def _buffer_ref_from_layer(layer: Any) -> list[BufferTensorRef]:
    buffer_address = _get_attr(layer, "buffer_address", default=None)
    module_address = _get_attr(layer, "module_address", default="") or ""
    if not buffer_address:
        return []
    if "." in buffer_address:
        module_path, buffer_name = buffer_address.rsplit(".", 1)
    else:
        module_path, buffer_name = module_address, buffer_address
    fq_name = f"{module_path}.{buffer_name}" if module_path else buffer_name
    return [BufferTensorRef(module_path=module_path, buffer_name=buffer_name, fq_name=fq_name)]


def _aggregation_kind(func_name: str) -> str | None:
    lowered = (func_name or "").lower()
    if lowered in {"cat", "stack"}:
        return lowered
    return None


def _indexing_metadata(layer: Any) -> Any:
    func_name = (_infer_func_name(layer) or "").lower()
    if func_name in {"getitem", "__getitem__"}:
        args = _layer_creation_args(layer)
        if len(args) >= 2:
            return args[1]
    if func_name in {"chunk", "split", "unbind"}:
        return {
            "index": _get_attr(layer, "iterable_output_index", default=None),
            "kwargs": _layer_creation_kwargs(layer),
        }
    return None


def _compute_depths(nodes: Mapping[str, GraphNode], topo: Sequence[str]) -> None:
    for label in topo:
        node = nodes[label]
        if not node.parent_labels:
            node.depth_from_input = 0
        else:
            node.depth_from_input = 1 + max(nodes[parent].depth_from_input for parent in node.parent_labels)

    for label in reversed(list(topo)):
        node = nodes[label]
        if not node.child_labels:
            node.depth_from_output = 0
        else:
            node.depth_from_output = 1 + max(nodes[child].depth_from_output for child in node.child_labels)


def identify_io_nodes(history: Any) -> tuple[list[str], list[str]]:
    layer_list = list(_get_attr(history, "layer_list", default=[])) or []
    input_labels = list(_get_attr(history, "input_layers", default=[])) or []
    output_labels = list(_get_attr(history, "output_layers", default=[])) or []

    if not input_labels:
        inferred_inputs: list[str] = []
        for layer in layer_list:
            label = _get_attr(layer, "layer_label", default=None)
            address = _get_attr(layer, "input_output_address", default=None)
            if label is None:
                continue
            if bool(_get_attr(layer, "is_input_layer", default=False)) or (
                isinstance(address, str) and address.startswith("input.")
            ):
                inferred_inputs.append(label)
        input_labels = inferred_inputs

    if not output_labels:
        inferred_outputs: list[str] = []
        for layer in layer_list:
            label = _get_attr(layer, "layer_label", default=None)
            address = _get_attr(layer, "input_output_address", default=None)
            if label is None:
                continue
            if bool(_get_attr(layer, "is_output_layer", default=False)) or (
                isinstance(address, str) and address.startswith("output.")
            ):
                inferred_outputs.append(label)
        output_labels = inferred_outputs

    if not output_labels:
        sinks: list[str] = []
        for layer in layer_list:
            label = _get_attr(layer, "layer_label", default=None)
            children = list(_get_attr(layer, "child_layers", default=[])) or []
            if label is not None and not children:
                sinks.append(label)
        output_labels = sinks
    return input_labels, output_labels


def build_parent_child_links(nodes: "OrderedDict[str, GraphNode]") -> None:
    for node in nodes.values():
        node.child_labels = []
    for node in nodes.values():
        for parent in node.parent_labels:
            if parent in nodes and node.label not in nodes[parent].child_labels:
                nodes[parent].child_labels.append(node.label)


def build_label_maps(history: Any) -> dict[str, int]:
    labels = list(_get_attr(history, "layer_labels", default=[])) or []
    return {label: index for index, label in enumerate(labels)}


def _topological_order_from_history(history: Any) -> list[str]:
    return list(_get_attr(history, "layer_labels", default=[])) or [
        layer.layer_label for layer in list(_get_attr(history, "layer_list", default=[]))
    ]


def _relevant_labels(nodes: Mapping[str, GraphNode], output_labels: Sequence[str]) -> list[str]:
    visited: set[str] = set()
    stack = list(output_labels)
    while stack:
        label = stack.pop()
        if label in visited or label not in nodes:
            continue
        visited.add(label)
        stack.extend(nodes[label].parent_labels)
    return [label for label in nodes if label in visited]


def _resolve_parameter_from_ref(model: torch.nn.Module, ref: ParameterTensorRef) -> torch.nn.Parameter | None:
    try:
        parameter = model.get_parameter(ref.fq_name)
    except Exception:
        try:
            module = _resolve_attr_path(model, ref.module_path)
            parameter = getattr(module, ref.param_name)
        except Exception:
            return None
    return parameter if isinstance(parameter, torch.nn.Parameter) else None


def _node_has_trainable_params(
    model: torch.nn.Module,
    param_refs: Sequence[ParameterTensorRef],
    *,
    fallback: bool,
    trainable_param_names: set[str] | None = None,
) -> bool:
    if trainable_param_names is not None:
        return any(ref.fq_name in trainable_param_names for ref in param_refs)
    for ref in param_refs:
        parameter = _resolve_parameter_from_ref(model, ref)
        if parameter is not None and parameter.requires_grad:
            return True
    return fallback


def _sample_args_kwargs(sample_input: Any, sample_kwargs: Mapping[str, Any] | None) -> tuple[tuple[Any, ...], dict[str, Any]]:
    kwargs = dict(sample_kwargs or {})
    if isinstance(sample_input, tuple):
        return sample_input, kwargs
    return (sample_input,), kwargs


def _build_input_spec(signature: inspect.Signature, args: tuple[Any, ...], kwargs: dict[str, Any]) -> TreeSpec:
    bound = normalize_runtime_inputs(signature, *args, partial=False, **kwargs)
    spec, _ = build_tree_spec(bound.arguments, "input")
    return spec


def build_graph_from_trace(
    model: torch.nn.Module,
    history: Any,
    sample_args: tuple[Any, ...],
    sample_kwargs: dict[str, Any],
    sample_output: Any,
    trainable_param_names: set[str] | None = None,
) -> GraphIR:
    layer_list = list(_get_attr(history, "layer_list", default=[])) or []
    parent_layer_lookup = {
        _get_attr(layer, "layer_label", default=f"layer_{index}"): layer
        for index, layer in enumerate(layer_list)
    }
    label2idx = build_label_maps(history)
    input_labels, output_labels = identify_io_nodes(history)
    topo = _topological_order_from_history(history)
    nodes: "OrderedDict[str, GraphNode]" = OrderedDict()

    for index, layer in enumerate(layer_list):
        label = _get_attr(layer, "layer_label", default=f"layer_{index}")
        explicit_parent_labels = list(_get_attr(layer, "parent_layers", default=[])) or []
        func_name = _infer_func_name(layer)
        node_type = (_get_attr(layer, "layer_type", default="operation") or "operation").lower()
        if label not in topo:
            topo.append(label)
        arg_template, kwarg_template, param_refs, inferred_parent_labels = _build_arg_templates(
            layer,
            previous_layers=layer_list[:index],
            parent_layer_lookup=parent_layer_lookup,
        )
        parent_labels = list(
            OrderedDict.fromkeys(explicit_parent_labels + [parent for parent in inferred_parent_labels if parent != label])
        )
        buffer_refs = _buffer_ref_from_layer(layer)
        tensor_shape = tuple(_get_attr(layer, "tensor_shape", default=()) or ()) or None
        tensor_dtype = _get_attr(layer, "tensor_dtype", default=None)
        numel = int(torch.tensor(tensor_shape).prod().item()) if tensor_shape else 0
        estimated_bytes = 0
        if tensor_dtype is not None and tensor_shape:
            estimated_bytes = int(numel * torch.tensor([], dtype=tensor_dtype).element_size())
        creation_args = _layer_creation_args(layer)
        containing_module = _get_attr(layer, "module_address", "containing_module_origin", default=None)
        node = GraphNode(
            label=label,
            node_type=node_type,
            func_name=func_name,
            func=_get_attr(layer, "func_applied", default=None),
            parent_labels=parent_labels,
            child_labels=list(_get_attr(layer, "child_layers", default=[])) or [],
            parent_arg_locations={
                "args": {
                    _normalize_path_key(path): parent
                    for path, parent in dict((_get_attr(layer, "parent_layer_arg_locs", default={}) or {}).get("args", {})).items()
                },
                "kwargs": {
                    _normalize_path_key(path): parent
                    for path, parent in dict((_get_attr(layer, "parent_layer_arg_locs", default={}) or {}).get("kwargs", {})).items()
                },
            },
            arg_template=arg_template,
            kwarg_template=kwarg_template,
            non_tensor_args={
                "position": list(_get_attr(layer, "func_position_args_non_tensor", default=[])) or [],
                "keyword": dict(_get_attr(layer, "func_keyword_args_non_tensor", default={}) or {}),
            },
            parameter_refs=param_refs,
            buffer_refs=buffer_refs,
            input_output_address=_get_attr(layer, "input_output_address", default=None),
            containing_module=containing_module,
            containing_modules=list(_get_attr(layer, "containing_modules_origin_nested", default=[])) or [],
            is_input=bool(_get_attr(layer, "is_input_layer", default=node_type == "input")),
            is_output=bool(_get_attr(layer, "is_output_layer", default=node_type == "output") or node_type == "output"),
            is_inplace=bool(_get_attr(layer, "func_is_inplace", default=False)),
            is_multi_output=bool(_get_attr(layer, "is_part_of_iterable_output", default=False)),
            multi_output_index=_get_attr(layer, "iterable_output_index", default=None),
            aggregation_kind=_aggregation_kind(func_name),
            indexing_metadata=_indexing_metadata(layer),
            tensor_shape=tensor_shape,
            tensor_dtype=tensor_dtype,
            numel=numel,
            estimated_bytes=estimated_bytes,
            estimated_flops=estimate_node_flops(node_type or func_name, tensor_shape, creation_args),
            has_trainable_params=_node_has_trainable_params(
                model,
                param_refs,
                fallback=bool(_get_attr(layer, "num_params_trainable", default=0)),
                trainable_param_names=trainable_param_names,
            ),
            topological_index=index,
        )
        nodes[label] = node

    build_parent_child_links(nodes)
    _compute_depths(nodes, topo)
    relevant_labels = _relevant_labels(nodes, output_labels)
    relevant_labels = [
        label
        for label in topo
        if label in relevant_labels and nodes[label].func_name.lower() not in SKIPPED_FUNCTION_NAMES
    ]

    signature = inspect.signature(model.forward)
    input_spec = _build_input_spec(signature, sample_args, sample_kwargs)
    output_spec, output_leaves = build_tree_spec(sample_output, "output")
    input_bound = normalize_runtime_inputs(signature, *sample_args, partial=False, **sample_kwargs)
    input_leaves = flatten_bound_inputs(input_bound)

    input_address_to_label = {}
    output_address_to_label = {}
    for label in input_labels:
        address = nodes[label].input_output_address
        if address is not None:
            input_address_to_label[address] = label
    for label in output_labels:
        address = nodes[label].input_output_address
        if address is not None:
            output_address_to_label[address] = label

    if not input_address_to_label and input_labels and input_leaves:
        for label, address in zip(input_labels, input_leaves.keys()):
            input_address_to_label[address] = label
    if not output_address_to_label and output_labels and output_leaves:
        output_address_to_label.update(
            _match_leaf_addresses_to_labels(output_labels, output_leaves, parent_layer_lookup)
        )
        if len(output_address_to_label) < min(len(output_labels), len(output_leaves)):
            assigned_labels = set(output_address_to_label.values())
            assigned_addresses = set(output_address_to_label.keys())
            remaining_labels = [label for label in output_labels if label not in assigned_labels]
            remaining_addresses = [
                address for address in output_leaves.keys() if address not in assigned_addresses
            ]
            for label, address in zip(remaining_labels, remaining_addresses):
                output_address_to_label[address] = label

    return GraphIR(
        nodes=nodes,
        input_labels=input_labels,
        output_labels=output_labels,
        topological_order=topo,
        relevant_labels=relevant_labels,
        input_spec=input_spec,
        output_spec=output_spec,
        input_address_to_label=input_address_to_label,
        output_address_to_label=output_address_to_label,
        forward_signature=signature,
        sample_args=sample_args,
        sample_kwargs=sample_kwargs,
        sample_input_spec=input_spec,
        sample_output_spec=output_spec,
    )


def trace_model(
    model: torch.nn.Module,
    sample_input: Any,
    sample_kwargs: Mapping[str, Any] | None = None,
) -> tuple[Any, tuple[Any, ...], dict[str, Any], Any]:
    if tl is None:  # pragma: no cover - guarded by caller
        raise ImportError("torchlens is required for graph tracing.")
    args, kwargs = _sample_args_kwargs(sample_input, sample_kwargs)
    with torch.no_grad():
        sample_output = model(*args, **kwargs)
    trace_kwargs = {
        "input_kwargs": kwargs or None,
        # TorchLens only captures the replay-critical per-layer call
        # arguments when activations are saved. We therefore keep the
        # exhaustive trace here, but immediately dispose of the raw
        # history once the graph IR has been built.
        "layers_to_save": "all",
        "keep_unsaved_layers": True,
        "save_function_args": True,
        "mark_input_output_distances": False,
        "detect_loops": False,
    }
    try:
        history = tl.log_forward_pass(model, args, **trace_kwargs)
    except Exception:
        tl.log_forward_pass(model, args, **trace_kwargs)
        history = tl.log_forward_pass(model, args, **trace_kwargs)
    return history, args, kwargs, sample_output
