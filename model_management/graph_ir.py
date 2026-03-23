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
    if isinstance(obj, list):
        return [_safe_deepcopy(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_safe_deepcopy(item) for item in obj)
    if isinstance(obj, dict):
        return {key: _safe_deepcopy(value) for key, value in obj.items()}
    try:
        return copy.deepcopy(obj)
    except Exception:
        return obj


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
    type_name: str | None = None


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
            return TreeSpec(kind="dataclass", type_name=type(current).__name__, children=fields)
        if isinstance(current, tuple) and hasattr(current, "_fields"):
            fields = OrderedDict()
            for field_name in current._fields:
                fields[field_name] = _build(getattr(current, field_name), f"{address}.{field_name}")
            return TreeSpec(kind="namedtuple", type_name=type(current).__name__, children=fields)
        if isinstance(current, list):
            return TreeSpec(
                kind="list",
                children=[_build(item, f"{address}.{index}") for index, item in enumerate(current)],
            )
        if isinstance(current, tuple):
            return TreeSpec(
                kind="tuple",
                children=[_build(item, f"{address}.{index}") for index, item in enumerate(current)],
            )
        if isinstance(current, dict):
            fields = OrderedDict()
            for key, value in current.items():
                fields[key] = _build(value, f"{address}.{key}")
            return TreeSpec(kind="dict", children=fields)
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
        return tuple(materialize_tree_spec(child, tensor_map) for child in spec.children)
    if spec.kind == "dict":
        return {key: materialize_tree_spec(child, tensor_map) for key, child in spec.children.items()}
    if spec.kind == "namedtuple":
        values = {key: materialize_tree_spec(child, tensor_map) for key, child in spec.children.items()}
        return values
    if spec.kind == "dataclass":
        values = {key: materialize_tree_spec(child, tensor_map) for key, child in spec.children.items()}
        return values
    raise ValueError(f"Unsupported TreeSpec kind: {spec.kind}")


def _replace_placeholders(obj: Any, resolver) -> Any:
    if isinstance(obj, (ParentTensorRef, ParameterTensorRef, BufferTensorRef, ConstantTensorRef)):
        return resolver(obj)
    if isinstance(obj, list):
        return [_replace_placeholders(item, resolver) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_replace_placeholders(item, resolver) for item in obj)
    if isinstance(obj, dict):
        return {key: _replace_placeholders(value, resolver) for key, value in obj.items()}
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
        if isinstance(value, list):
            for index, item in enumerate(value):
                _walk(item, f"{prefix}.{index}")
            return
        if isinstance(value, tuple):
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
        available_tensors: Mapping[str, torch.Tensor],
        model: torch.nn.Module,
        device: torch.device | None = None,
    ) -> tuple[list[Any], dict[str, Any]]:
        def _resolve(ref: ArgPlaceholder) -> Any:
            if isinstance(ref, ParentTensorRef):
                return available_tensors[ref.parent_label]
            if isinstance(ref, ParameterTensorRef):
                module = model.get_submodule(ref.module_path) if ref.module_path else model
                return getattr(module, ref.param_name)
            if isinstance(ref, BufferTensorRef):
                module = model.get_submodule(ref.module_path) if ref.module_path else model
                try:
                    return module.get_buffer(ref.buffer_name)
                except AttributeError:
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


def _build_arg_templates(layer: Any) -> tuple[list[Any], dict[str, Any], list[ParameterTensorRef]]:
    creation_args = _layer_creation_args(layer)
    creation_kwargs = _layer_creation_kwargs(layer)

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

    def _mark(obj: Any, location_kind: str, prefix: tuple[Any, ...] = ()) -> Any:
        if isinstance(obj, torch.Tensor):
            parent_label = parent_arg_locations[location_kind].get(prefix)
            if parent_label is not None:
                return ParentTensorRef(parent_label)
            if param_queue:
                return param_queue.pop(0)
            return ConstantTensorRef(obj.detach().cpu())
        if isinstance(obj, list):
            return [_mark(item, location_kind, prefix + (index,)) for index, item in enumerate(obj)]
        if isinstance(obj, tuple):
            return tuple(_mark(item, location_kind, prefix + (index,)) for index, item in enumerate(obj))
        if isinstance(obj, dict):
            return {
                key: _mark(value, location_kind, prefix + (key,))
                for key, value in obj.items()
            }
        return _safe_deepcopy(obj)

    arg_template = _mark(creation_args, "args", ())
    kwarg_template = _mark(creation_kwargs, "kwargs", ())
    return arg_template, kwarg_template, param_refs


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
) -> GraphIR:
    layer_list = list(_get_attr(history, "layer_list", default=[])) or []
    label2idx = build_label_maps(history)
    input_labels, output_labels = identify_io_nodes(history)
    topo = _topological_order_from_history(history)
    nodes: "OrderedDict[str, GraphNode]" = OrderedDict()

    for index, layer in enumerate(layer_list):
        label = _get_attr(layer, "layer_label", default=f"layer_{index}")
        parent_labels = list(_get_attr(layer, "parent_layers", default=[])) or []
        func_name = _infer_func_name(layer)
        node_type = (_get_attr(layer, "layer_type", default="operation") or "operation").lower()
        if label not in topo:
            topo.append(label)
        arg_template, kwarg_template, param_refs = _build_arg_templates(layer)
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
            is_multi_output=bool(_get_attr(layer, "is_part_of_iterable_output", default=False)),
            multi_output_index=_get_attr(layer, "iterable_output_index", default=None),
            aggregation_kind=_aggregation_kind(func_name),
            indexing_metadata=_indexing_metadata(layer),
            tensor_shape=tensor_shape,
            tensor_dtype=tensor_dtype,
            numel=numel,
            estimated_bytes=estimated_bytes,
            estimated_flops=estimate_node_flops(node_type or func_name, tensor_shape, creation_args),
            has_trainable_params=bool(_get_attr(layer, "num_params_trainable", default=0)),
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
        for label, address in zip(output_labels, output_leaves.keys()):
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
    try:
        history = tl.log_forward_pass(
            model,
            args,
            input_kwargs=kwargs or None,
            layers_to_save="all",
            keep_unsaved_layers=True,
            save_function_args=True,
            mark_input_output_distances=False,
        )
    except Exception:
        tl.log_forward_pass(model, args, input_kwargs=kwargs or None)
        history = tl.log_forward_pass(
            model,
            args,
            input_kwargs=kwargs or None,
            layers_to_save="all",
            keep_unsaved_layers=True,
            save_function_args=True,
            mark_input_output_distances=False,
        )
    return history, args, kwargs, sample_output
