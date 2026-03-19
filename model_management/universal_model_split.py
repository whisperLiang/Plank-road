"""
Universal Model Splitting for Split-Learning Continual Learning
================================================================

Splits **any** PyTorch model at an **arbitrary layer** for edge-cloud
split inference and training.

The core idea comes from two references:

* **TorchLens** (``torchlens``) — traces a model's full computational graph
  operation-by-operation, producing a ``ModelHistory`` / list of
  ``TensorLogEntry`` objects that can be replayed layer by layer.
* **Shawarma** ``model_utils.py`` — demonstrates how that layer list can be
  partitioned at any index, with the head running on the edge and the tail
  on the cloud, for both inference and training.

Split-point selection profile is generalized for arbitrary models:

``profile_layers()`` automatically computes FLOPs, tensor size, 

and privacy-leakage estimates into ``LayerProfile``.

This module adapts both ideas into a single ``UniversalModelSplitter`` class
that integrates with the existing Plank-road edge-cloud continual-learning
pipeline while remaining completely model-agnostic.

Key capabilities
-----------------
* ``trace(model, sample_input)`` — one-time call that captures the full
  graph of layer operations, parameters, buffers, and non-tensor arguments.
* ``list_layers()`` — returns a human-readable list of all traceable layers,
  making it easy for a user to pick a split point.
* ``split(layer_index)`` — designates a split point; after this the
  ``edge_forward`` / ``cloud_forward`` / ``cloud_train_step`` helpers are
  available.
* ``edge_forward(x)`` — runs the head partition (layers 0..split) and
  returns intermediate activations.
* ``cloud_forward(intermediate)`` — runs the tail partition on the cloud
  and returns the model output.
* ``cloud_train_step(intermediate, ...)`` — forward + backward on the
  cloud tail; returns loss dict.
* Serialisation helpers for transmitting intermediate tensors over the
  network (``serialise_intermediate`` / ``deserialise_intermediate``).

The class does **not** depend on the model being a Faster R-CNN or any
particular architecture.  Any ``nn.Module`` that TorchLens can trace (which
includes VGG, ResNet, ViT, Swin, DenseNet, etc.) is supported.

Usage example
-------------
>>> import torch, torchvision
>>> from model_management.universal_model_split import UniversalModelSplitter
>>>
>>> model = torchvision.models.resnet18(pretrained=True)
>>> splitter = UniversalModelSplitter()
>>> splitter.trace(model, torch.rand(1, 3, 224, 224))
>>> splitter.list_layers()          # inspect available split points
>>> splitter.split(layer_index=15)  # split after the 15th operation
>>>
>>> # Edge side
>>> inter = splitter.edge_forward(real_input)
>>> data  = splitter.serialise_intermediate(inter)
>>>
>>> # Cloud side (received over network)
>>> inter = splitter.deserialise_intermediate(data)
>>> output = splitter.cloud_forward(inter)
"""

from __future__ import annotations

import copy
import io
import os
import pickle
import random
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

# ---------------------------------------------------------------------------
# Required dependencies.
# ---------------------------------------------------------------------------
import torchlens as tl
from model_management.activation_sparsity import (
    DASTrainer,
    apply_das_to_tail,
)

_HAS_TORCHLENS = True
_HAS_DAS = True


def _move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, list):
        return [_move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_device(v, device) for v in obj)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj


def _detach_clone_any(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    if isinstance(obj, list):
        return [_detach_clone_any(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_detach_clone_any(v) for v in obj)
    if isinstance(obj, dict):
        return {k: _detach_clone_any(v) for k, v in obj.items()}
    return obj


def _extract_first_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (list, tuple)):
        for v in obj:
            t = _extract_first_tensor(v)
            if t is not None:
                return t
        return None
    if isinstance(obj, dict):
        for v in obj.values():
            t = _extract_first_tensor(v)
            if t is not None:
                return t
        return None
    return None


def _is_proxy_split_model(model: nn.Module) -> bool:
    mod = getattr(model.__class__, "__module__", "")
    if mod.startswith("ultralytics"):
        return True
    if mod.startswith("torchvision.models.detection"):
        return True
    if hasattr(model, "rpn") and hasattr(model, "roi_heads"):
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Helper: build label→index map and prepare layers for replay
# ═══════════════════════════════════════════════════════════════════════════

def _prepare_replay_graph(layer_list: list) -> dict[str, int]:
    """Build a ``{layer_label: index}`` mapping for the layer list."""
    label2idx: dict[str, int] = {}
    for i, layer in enumerate(layer_list):
        label2idx[layer.layer_label] = i
    return label2idx


def _layer_get_activation(layer):
    act = getattr(layer, "activation", None)
    if act is not None:
        return act
    return getattr(layer, "tensor_contents", None)


def _layer_set_activation(layer, value) -> None:
    setattr(layer, "activation", value)
    if hasattr(layer, "tensor_contents"):
        setattr(layer, "tensor_contents", value)


def _layer_func_name(layer) -> str:
    name = getattr(layer, "func_name", None)
    if name:
        return name
    return getattr(layer, "func_applied_name", "") or ""


def _layer_non_tensor_kwargs(layer) -> dict:
    kw = getattr(layer, "func_kwargs_non_tensor", None)
    if isinstance(kw, dict):
        return dict(kw)
    old_kw = getattr(layer, "func_keyword_args_non_tensor", None)
    if isinstance(old_kw, dict):
        return dict(old_kw)
    return {}


def _is_non_splittable_layer(layer) -> bool:
    """Return True for layers that should never be chosen as split points."""
    layer_type = str(getattr(layer, "layer_type", "") or "").lower()
    label = str(getattr(layer, "layer_label", "") or "").lower()
    if layer_type in {"input", "output", "buffer"}:
        return True
    if label.startswith("buffer_"):
        return True
    return False


def _nearest_splittable_index(layer_list: list, requested_index: int) -> int | None:
    """Find the nearest valid split index around requested_index.

    Preference order: closest previous layer, then closest next layer.
    """
    n = len(layer_list)
    if n < 3:
        return None

    lo = 1
    hi = n - 2
    req = max(lo, min(requested_index, hi))

    if not _is_non_splittable_layer(layer_list[req]):
        return req

    for delta in range(1, max(req - lo, hi - req) + 1):
        left = req - delta
        if left >= lo and not _is_non_splittable_layer(layer_list[left]):
            return left
        right = req + delta
        if right <= hi and not _is_non_splittable_layer(layer_list[right]):
            return right
    return None


def _collect_tail_boundary_labels(
    layer_list: list,
    label2idx: dict[str, int],
    split_index: int,
) -> set[str]:
    """Collect head-side layer labels required by tail-side replay.

    For DAG models (e.g. residual/skip connections), layers in the tail can
    depend on multiple parent activations from the head partition. This helper
    finds all such cross-boundary parent labels.
    """
    needed: set[str] = set()
    for idx in range(split_index + 1, len(layer_list)):
        layer = layer_list[idx]
        arg_locs = getattr(layer, "parent_layer_arg_locs", None) or {"args": {}, "kwargs": {}}

        parent_labels = []
        parent_labels.extend(list(arg_locs.get("args", {}).values()))
        parent_labels.extend(list(arg_locs.get("kwargs", {}).values()))

        for plabel in parent_labels:
            pidx = label2idx.get(plabel)
            if pidx is not None and pidx <= split_index:
                needed.add(plabel)
    return needed


def _estimate_layer_flops(layer) -> float:
    """Heuristic FLOPs estimate for one traced layer."""
    fn = _layer_func_name(layer).lower()
    shape = getattr(layer, "tensor_shape", None)
    num_params = float(getattr(layer, "num_params_total", 0) or 0)

    if "conv" in fn:
        if shape and len(shape) >= 4:
            _, _, h, w = shape[:4]
            return max(2.0 * num_params * max(h * w, 1), 0.0)
        return max(2.0 * num_params, 0.0)

    if "linear" in fn or fn in ("mm", "addmm", "matmul"):
        return max(2.0 * num_params, 0.0)

    if shape:
        return float(np.prod(shape)) * 0.1
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Helper: layer-by-layer forward pass (full model, eval mode)
# ═══════════════════════════════════════════════════════════════════════════

def _replay_layer(layer, layer_list, label2idx, fallback_input=None,
                  model: torch.nn.Module | None = None):
    """Execute one traced layer using the latest torchlens LayerPassLog API."""
    func = layer.func_applied
    if func is None:
        tc = _layer_get_activation(layer)
        return tc if tc is not None else fallback_input

    arg_locs = getattr(layer, "parent_layer_arg_locs", None) or {"args": {}, "kwargs": {}}

    flat_parent_locs: dict = {}
    nested_parent_locs: dict = {}
    for key, plabel in arg_locs.get("args", {}).items():
        if isinstance(key, tuple):
            nested_parent_locs[key] = plabel
        else:
            pos = int(key) if isinstance(key, str) else key
            flat_parent_locs[pos] = plabel

    parent_positions = set(flat_parent_locs.keys())
    for (pos, _sub) in nested_parent_locs:
        parent_positions.add(pos)

    param_name_to_tensor: dict[str, torch.Tensor] = {}
    if model is not None:
        named = dict(model.named_parameters())
        for pl in (getattr(layer, "parent_param_logs", None) or []):
            key = f"{pl.module_address}.{pl.name}"
            if key in named:
                param_name_to_tensor[pl.name] = named[key]

    func_name = _layer_func_name(layer)
    argnames = list(getattr(layer, "func_argnames", ()) or ())
    non_tensor_args = list(getattr(layer, "func_positional_args_non_tensor", []) or [])
    if not non_tensor_args:
        non_tensor_args = list(getattr(layer, "func_non_tensor_args", []) or [])

    if func_name == "batch_norm":
        creation_args = [None] * 9
        p0 = flat_parent_locs.get(0)
        if p0 in label2idx:
            creation_args[0] = _layer_get_activation(layer_list[label2idx[p0]])
        elif fallback_input is not None:
            creation_args[0] = fallback_input

        creation_args[1] = param_name_to_tensor.get("weight")
        creation_args[2] = param_name_to_tensor.get("bias")

        p3 = flat_parent_locs.get(3)
        if p3 in label2idx:
            creation_args[3] = _layer_get_activation(layer_list[label2idx[p3]])
        p4 = flat_parent_locs.get(4)
        if p4 in label2idx:
            creation_args[4] = _layer_get_activation(layer_list[label2idx[p4]])

        creation_args[5] = non_tensor_args[0] if len(non_tensor_args) > 0 else False
        creation_args[6] = non_tensor_args[1] if len(non_tensor_args) > 1 else 0.1
        creation_args[7] = non_tensor_args[2] if len(non_tensor_args) > 2 else 1e-5
        creation_args[8] = non_tensor_args[3] if len(non_tensor_args) > 3 else True
    elif argnames:
        creation_args = []
        non_tensor_iter = iter(non_tensor_args)
        for i, argname in enumerate(argnames):
            if i in flat_parent_locs:
                creation_args.append(None)
            elif argname in param_name_to_tensor:
                creation_args.append(param_name_to_tensor[argname])
            else:
                creation_args.append(next(non_tensor_iter, None))

        # torchlens may store reshape/view target shape as multiple
        # positional non-tensor args while func_argnames has a single
        # "shape" or "size" slot.
        if (
            func_name in {"reshape", "view"}
            and len(argnames) >= 2
            and argnames[1] in {"shape", "size"}
            and len(non_tensor_args) > 1
        ):
            creation_args[1] = tuple(non_tensor_args)
    else:
        creation_args = list(getattr(layer, "creation_args", []) or non_tensor_args)

    for pos, plabel in flat_parent_locs.items():
        if plabel in label2idx:
            parent = layer_list[label2idx[plabel]]
            tc = _layer_get_activation(parent)
            while len(creation_args) <= pos:
                creation_args.append(None)
            if tc is not None:
                creation_args[pos] = tc
            elif fallback_input is not None:
                creation_args[pos] = fallback_input

    if nested_parent_locs:
        nested_groups: dict[int, dict[int, str]] = {}
        for (pos, sub), plabel in nested_parent_locs.items():
            nested_groups.setdefault(pos, {})[sub] = plabel
        for pos, sub_map in nested_groups.items():
            elems: list = []
            for sub_idx in sorted(sub_map.keys()):
                plabel = sub_map[sub_idx]
                if plabel in label2idx:
                    parent = layer_list[label2idx[plabel]]
                    tc = _layer_get_activation(parent)
                    elems.append(tc if tc is not None else fallback_input)
                else:
                    elems.append(fallback_input)
            if pos < len(creation_args):
                creation_args[pos] = elems
            else:
                while len(creation_args) <= pos:
                    creation_args.append(None)
                creation_args[pos] = elems

    kw_args = _layer_non_tensor_kwargs(layer)
    for kw_name, plabel in arg_locs.get("kwargs", {}).items():
        if plabel in label2idx:
            parent = layer_list[label2idx[plabel]]
            tc = _layer_get_activation(parent)
            if tc is not None:
                kw_args[kw_name] = tc

    if argnames:
        positional_names = set(argnames[:len(creation_args)])
        for name in list(kw_args.keys()):
            if name in positional_names:
                kw_args.pop(name, None)

    if func_name.endswith("_") and model is not None:
        creation_args = [
            a.clone() if isinstance(a, torch.Tensor) and a.is_leaf and a.requires_grad else a
            for a in creation_args
        ]

    while creation_args and creation_args[-1] is None:
        creation_args.pop()

    result = func(*creation_args, **kw_args)

    # Handle multi-output functions (chunk, split, unbind, etc.)
    # torchlens creates one layer per tuple element; extract the right one.
    if isinstance(result, (tuple, list)):
        label_short = getattr(layer, "layer_label_short", "") or ""
        parts = label_short.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            out_idx = int(parts[1]) - 1
            if 0 <= out_idx < len(result):
                return result[out_idx]
        return result[0] if result else result

    return result


def _layer_forward_full(
    layer_list: list,
    label2idx: dict[str, int],
    x: torch.Tensor,
    model: torch.nn.Module | None = None,
) -> torch.Tensor:
    """Execute the full model layer-by-layer via traced graph replay.

    Parameters
    ----------
    layer_list : list of TensorLog entries
    label2idx  : dict mapping layer_label → list index
    x          : input tensor

    Returns
    -------
    torch.Tensor — model output
    """
    for layer in layer_list:
        func_name = _layer_func_name(layer)
        layer_type = getattr(layer, "layer_type", None)

        if func_name == "none":
            if layer_type == "input":
                _layer_set_activation(layer, x)
            elif layer_type == "output":
                parent_idx = label2idx[layer.parent_layers[0]]
                _layer_set_activation(layer, _layer_get_activation(layer_list[parent_idx]))
            continue

        result = _replay_layer(layer, layer_list, label2idx, fallback_input=x, model=model)
        _layer_set_activation(layer, result)
        x = result

    return x


# ═══════════════════════════════════════════════════════════════════════════
# Helper: partition forward (head / tail)
# ═══════════════════════════════════════════════════════════════════════════

def _partition_forward_head(
    layer_list: list,
    label2idx: dict[str, int],
    x: torch.Tensor,
    split_index: int,
    model: torch.nn.Module | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run layers ``[0 .. split_index]`` and return ``(final_output, intermediate)``."""
    intermediate: torch.Tensor | None = None

    for idx, layer in enumerate(layer_list):
        if idx > split_index:
            break

        func_name = _layer_func_name(layer)
        layer_type = getattr(layer, "layer_type", None)

        if func_name == "none":
            if layer_type == "input":
                _layer_set_activation(layer, x)
            elif layer_type == "output" and layer.parent_layers:
                parent_idx = label2idx[layer.parent_layers[0]]
                _layer_set_activation(layer, _layer_get_activation(layer_list[parent_idx]))
            if idx == split_index:
                tc = _layer_get_activation(layer)
                intermediate = tc.detach().clone() if tc is not None else x.detach().clone()
            continue

        result = _replay_layer(layer, layer_list, label2idx, fallback_input=x, model=model)
        _layer_set_activation(layer, result)
        x = result

        if idx == split_index:
            intermediate = x.detach().clone()

    if intermediate is None:
        intermediate = x.detach().clone()
    return x, intermediate


def _partition_forward_tail(
    layer_list: list,
    label2idx: dict[str, int],
    x_inter: torch.Tensor,
    split_index: int,
    model: torch.nn.Module | None = None,
) -> torch.Tensor:
    """Run layers ``[split_index+1 .. end]`` given the intermediate tensor."""
    _layer_set_activation(layer_list[split_index], x_inter)
    x = x_inter

    for idx, layer in enumerate(layer_list):
        if idx <= split_index:
            continue

        func_name = _layer_func_name(layer)
        layer_type = getattr(layer, "layer_type", None)

        if func_name == "none":
            if layer_type == "output" and layer.parent_layers:
                plabel = layer.parent_layers[0]
                if plabel in label2idx and label2idx[plabel] > split_index:
                    _layer_set_activation(layer, _layer_get_activation(layer_list[label2idx[plabel]]))
                else:
                    _layer_set_activation(layer, x)
            continue

        result = _replay_layer(layer, layer_list, label2idx, fallback_input=x_inter, model=model)
        _layer_set_activation(layer, result)
        x = result

    return x


def _partition_forward_tail_train(
    layer_list: list,
    label2idx: dict[str, int],
    x_inter: torch.Tensor,
    split_index: int,
    model: torch.nn.Module | None = None,
) -> torch.Tensor:
    """Training-mode forward through the tail partition (keeps autograd graph)."""
    _layer_set_activation(layer_list[split_index], x_inter)
    x = x_inter

    for idx, layer in enumerate(layer_list):
        if idx <= split_index:
            continue

        func_name = _layer_func_name(layer)
        layer_type = getattr(layer, "layer_type", None)

        if func_name == "none":
            if layer_type == "output" and layer.parent_layers:
                plabel = layer.parent_layers[0]
                if plabel in label2idx and label2idx[plabel] > split_index:
                    _layer_set_activation(layer, _layer_get_activation(layer_list[label2idx[plabel]]))
                else:
                    _layer_set_activation(layer, x)
            continue

        result = _replay_layer(layer, layer_list, label2idx,
                               fallback_input=x_inter, model=model)
        _layer_set_activation(layer, result)
        x = result

    return x


# ═══════════════════════════════════════════════════════════════════════════
# Helper: freeze / unfreeze parameters by partition index
# ═══════════════════════════════════════════════════════════════════════════

def _get_param_tensors_for_layer(layer, model: torch.nn.Module) -> list[torch.Tensor]:
    """Return actual nn.Parameter tensors associated with a traced layer.

    In torchlens >= 0.12 the actual tensors are not stored on the log entry.
    We resolve ``parent_param_logs`` metadata back to the original model via
    ``model.named_parameters()`` using ``module_address`` + ``name``.
    """
    param_logs = getattr(layer, "parent_param_logs", None)
    if not param_logs:
        return []
    named = dict(model.named_parameters())
    params: list[torch.Tensor] = []
    for pl in param_logs:
        key = f"{pl.module_address}.{pl.name}"
        if key in named:
            params.append(named[key])
    return params


def _freeze_head_params(layer_list: list, split_index: int, model: torch.nn.Module) -> None:
    """Freeze all parameters in layers ``[0 .. split_index]``."""
    limit = min(split_index + 1, len(layer_list))
    for i in range(limit):
        for param in _get_param_tensors_for_layer(layer_list[i], model):
            param.requires_grad = False


def _unfreeze_tail_params(layer_list: list, split_index: int, model: torch.nn.Module) -> None:
    """Ensure all parameters in layers ``[split_index+1 .. end]`` are trainable."""
    for i in range(split_index + 1, len(layer_list)):
        for param in _get_param_tensors_for_layer(layer_list[i], model):
            param.requires_grad = True


# ═══════════════════════════════════════════════════════════════════════════
# Helper: parameter access helpers
# ═══════════════════════════════════════════════════════════════════════════

def _get_all_params(layer_list: list, model: torch.nn.Module, device=None) -> list[torch.Tensor]:
    """Flatten and collect all parameters across all layers."""
    seen: set[int] = set()
    params: list[torch.Tensor] = []
    for layer in layer_list:
        for p in _get_param_tensors_for_layer(layer, model):
            pid = id(p)
            if pid not in seen:
                seen.add(pid)
                params.append(p if device is None else p.to(device))
    return params


def _get_trainable_params(layer_list: list, split_index: int, model: torch.nn.Module) -> list[torch.Tensor]:
    """Return all trainable (``requires_grad=True``) params in the tail partition."""
    seen: set[int] = set()
    params: list[torch.Tensor] = []
    for i in range(split_index + 1, len(layer_list)):
        for p in _get_param_tensors_for_layer(layer_list[i], model):
            pid = id(p)
            if pid not in seen and p.requires_grad:
                seen.add(pid)
                params.append(p)
    return params


def _load_weights_into_layers(layer_list: list, model: torch.nn.Module, weights: list[torch.Tensor]) -> None:
    """Load weights sequentially into the layer list."""
    dst_params: list[torch.Tensor] = []
    for layer in layer_list:
        dst_params.extend(_get_param_tensors_for_layer(layer, model))

    for p, w in zip(dst_params, weights):
        p.data.copy_(w)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: set train / eval mode on the layer list
# ═══════════════════════════════════════════════════════════════════════════

def _set_training_flag(layer_list: list, training: bool) -> None:
    """Set the ``training`` non-tensor arg to *training* for layers that
    have it (e.g. Dropout, BatchNorm).  Mirrors Shawarma ``train_mode`` /
    ``eval_mode``.
    """
    for layer in layer_list:
        argnames = getattr(layer, "func_argnames", None)
        if argnames and "training" in argnames:
            kwargs = getattr(layer, "func_kwargs_non_tensor", None)
            if isinstance(kwargs, dict) and "training" in kwargs:
                kwargs["training"] = training


# ═══════════════════════════════════════════════════════════════════════════
# Data class for layer info returned to the user
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LayerInfo:
    """Lightweight summary of a traced layer — returned by ``list_layers()``."""
    index: int
    label: str
    layer_type: str
    func_name: str
    output_shape: Optional[tuple] = None
    has_params: bool = False
    num_params: int = 0
    module_name: Optional[str] = None

    def __repr__(self) -> str:
        s = f"[{self.index:>3d}] {self.label:<30s}  type={self.layer_type}"
        if self.func_name and self.func_name != "none":
            s += f"  fn={self.func_name}"
        if self.has_params:
            s += f"  params={self.num_params}"
        if self.output_shape:
            s += f"  out_shape={self.output_shape}"
        if self.module_name:
            s += f"  module={self.module_name}"
        return s


# ═══════════════════════════════════════════════════════════════════════════
# Layer profiling data
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LayerProfile:
    """Per-candidate-split-point cost profile.

    This mirrors the ``layer_info`` dict from previous ``config.py``::

        layer_info = { action_idx : [cumulated_flops, smashed_data_size], ... }

    We extend it with a ``privacy_leakage`` estimate (higher → more info
    leaked in the intermediate tensor).

    Attributes
    ----------
    index : int
        Split-point layer index in the traced graph.
    label : str
        Human-readable layer label.
    cumulative_flops : float
        Estimated total FLOPs for layers 0..index.
    smashed_data_size : int
        Number of scalar elements in the intermediate tensor (``numel``).
    output_shape : tuple | None
        Shape of the activation tensor at this layer.
    privacy_leakage : float
        Heuristic privacy cost ∈ [0, 1].  Larger intermediates leak more.
    """
    index: int
    label: str
    cumulative_flops: float = 0.0
    smashed_data_size: int = 0
    output_shape: Optional[tuple] = None
    privacy_leakage: float = 0.0

    def __repr__(self) -> str:
        return (
            f"LayerProfile(idx={self.index}, label={self.label!r}, "
            f"flops={self.cumulative_flops:.0f}, "
            f"smashed={self.smashed_data_size}, "
            f"privacy={self.privacy_leakage:.4f})"
        )


class SplitPointSelector:
    """Simple selector over profiled split candidates."""

    def __init__(self, profiles: List[LayerProfile]):
        if not profiles:
            raise ValueError("profiles must not be empty")
        self._profiles = list(profiles)

    def select(self) -> int:
        return self.select_by_flops_ratio(0.5)

    def select_by_flops_ratio(self, target_ratio: float = 0.5) -> int:
        total = max(self._profiles[-1].cumulative_flops, 1.0)
        target = max(0.0, min(1.0, target_ratio)) * total
        best = min(self._profiles, key=lambda p: abs(p.cumulative_flops - target))
        return best.index

    def select_by_privacy(self, max_leakage: float = 0.5) -> int:
        valid = [p for p in self._profiles if p.privacy_leakage <= max_leakage]
        if valid:
            return valid[-1].index
        return min(self._profiles, key=lambda p: p.privacy_leakage).index

    def get_profiles(self) -> List[LayerProfile]:
        return list(self._profiles)


# ═══════════════════════════════════════════════════════════════════════════
# Main class
# ═══════════════════════════════════════════════════════════════════════════

class UniversalModelSplitter:
    """Model-agnostic splitter for edge-cloud split learning.

    Wraps TorchLens model tracing so that *any* ``nn.Module`` can be split
    at an arbitrary layer (chosen by index or label) and executed in two
    separate partitions: an **edge head** and a **cloud tail**.

    Typical workflow::

        splitter = UniversalModelSplitter()
        splitter.trace(model, sample_input)
        splitter.list_layers()             # pick your split point
        splitter.split(layer_index=15)

        # Edge
        inter = splitter.edge_forward(real_input)
        data  = splitter.serialise_intermediate(inter)

        # Cloud
        inter = splitter.deserialise_intermediate(data)
        out   = splitter.cloud_forward(inter)

    Parameters
    ----------
    device : torch.device | str | None
        Default device for computation.  Inferred from the model if ``None``.
    """

    def __init__(self, device: torch.device | str | None = None):
        self.device = torch.device(device) if isinstance(device, str) else device
        self._model: nn.Module | None = None
        self._model_history = None  # torchlens ModelHistory
        self._layer_list: list | None = None  # list[TensorLogEntry]
        self._label2idx: dict[str, int] = {}
        self._split_index: int | None = None
        self._traced = False
        self._use_proxy_split = False
        self._proxy_input: Any | None = None

    # ------------------------------------------------------------------
    # 1. Tracing
    # ------------------------------------------------------------------

    def trace(
        self,
        model: nn.Module,
        sample_input: torch.Tensor | list | tuple,
        *,
        save_function_args: bool = True,
    ) -> "UniversalModelSplitter":
        """Trace the model's computational graph using TorchLens.

        This must be called once before any splitting / forwarding.

        Parameters
        ----------
        model : nn.Module
            The PyTorch model to trace.
        sample_input : Tensor | list | tuple
            A representative input (shape and dtype must match real inputs).
        save_function_args : bool
            Whether TorchLens should save non-tensor function arguments
            (needed for faithful replay).

        Returns
        -------
        self — for method chaining.
        """
        if not _HAS_TORCHLENS:
            raise ImportError(
                "torchlens is required for universal model splitting. "
                "Install it with:  pip install torchlens"
            )

        self._model = model
        if self.device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

        model.eval()
        with torch.no_grad():
            mh = tl.log_forward_pass(
                model,
                sample_input,
                layers_to_save="all",
                save_function_args=save_function_args,
            )

        self._model_history = mh
        self._layer_list = list(mh.layer_list)  # shallow copy of the list
        self._label2idx = _prepare_replay_graph(self._layer_list)
        self._traced = True
        self._split_index = None
        self._proxy_input = None

        # Some model families are not reliably replayable op-by-op across
        # versions. In proxy mode we still allow arbitrary logical split points,
        # while cloud/head execution uses native model forward.
        self._use_proxy_split = _is_proxy_split_model(model)

        logger.info(
            "[UniversalSplit] Traced {} — {} layer operations captured.",
            mh.model_name,
            len(self._layer_list),
        )
        return self

    # ------------------------------------------------------------------
    # 2. Inspection
    # ------------------------------------------------------------------

    def list_layers(self) -> list[LayerInfo]:
        """Return a list of ``LayerInfo`` summarising every traced layer.

        Useful for deciding where to split.
        """
        self._ensure_traced()
        infos: list[LayerInfo] = []
        for i, layer in enumerate(self._layer_list):
            num_params = int(getattr(layer, "num_params_total", 0) or 0)
            num_param_tensors = int(getattr(layer, "num_param_tensors", 0) or 0)
            has_params = bool(num_param_tensors > 0 or num_params > 0)
            infos.append(
                LayerInfo(
                    index=i,
                    label=getattr(layer, "layer_label", f"layer_{i}"),
                    layer_type=getattr(layer, "layer_type", "unknown"),
                    func_name=_layer_func_name(layer),
                    output_shape=getattr(layer, "tensor_shape", None),
                    has_params=has_params,
                    num_params=num_params,
                    module_name=getattr(layer, "containing_module_origin", None),
                )
            )
        return infos

    def num_layers(self) -> int:
        self._ensure_traced()
        return len(self._layer_list)

    def get_layer_info(self, index_or_label: int | str) -> LayerInfo:
        """Get info for a single layer (by index or label string)."""
        self._ensure_traced()
        if isinstance(index_or_label, str):
            idx = self._label2idx.get(index_or_label)
            if idx is None:
                raise KeyError(f"Layer label '{index_or_label}' not found in traced graph.")
        else:
            idx = index_or_label
        return self.list_layers()[idx]

    # ------------------------------------------------------------------
    # 3. Splitting
    # ------------------------------------------------------------------

    def split(self, layer_index: int | None = None, layer_label: str | None = None) -> "UniversalModelSplitter":
        """Designate the split point.

        Everything up to and including ``layer_index`` runs on the **edge**;
        everything after runs on the **cloud**.

        Parameters
        ----------
        layer_index : int | None
            Ordinal index of the split layer (0-based).  Mutually exclusive
            with *layer_label*.
        layer_label : str | None
            Human-readable label (e.g. ``'conv2d_3_7'``).

        Returns
        -------
        self — for chaining.
        """
        self._ensure_traced()
        if layer_index is not None and layer_label is not None:
            raise ValueError("Specify exactly one of layer_index or layer_label, not both.")
        if layer_label is not None:
            layer_index = self._label2idx.get(layer_label)
            if layer_index is None:
                raise KeyError(f"Layer label '{layer_label}' not found.")
        if layer_index is None:
            raise ValueError("Either layer_index or layer_label is required.")
        if layer_index < 0 or layer_index >= len(self._layer_list):
            raise IndexError(
                f"layer_index {layer_index} out of range [0, {len(self._layer_list) - 1}]"
            )

        resolved = _nearest_splittable_index(self._layer_list, layer_index)
        if resolved is None:
            raise ValueError("No valid split point found: graph only contains non-splittable layers.")
        if resolved != layer_index:
            logger.warning(
                "[UniversalSplit] Requested split index {} is non-splittable ({}); "
                "using nearest valid index {} ({}).",
                layer_index,
                getattr(self._layer_list[layer_index], "layer_label", f"layer_{layer_index}"),
                resolved,
                getattr(self._layer_list[resolved], "layer_label", f"layer_{resolved}"),
            )
        layer_index = resolved

        self._split_index = layer_index
        logger.info(
            "[UniversalSplit] Split point set at layer {} (index {}).",
            self._layer_list[layer_index].layer_label,
            layer_index,
        )
        return self

    @property
    def split_index(self) -> int | None:
        return self._split_index

    def find_best_split_by_module(self, module_name: str) -> int:
        """Find the last layer that belongs to *module_name* and return its
        index.  Useful for splitting at module boundaries (e.g.
        ``'features'``, ``'backbone'``, ``'encoder'``).
        """
        self._ensure_traced()
        best = -1
        for i, layer in enumerate(self._layer_list):
            mods = getattr(layer, "modules_exited", []) or []
            containing = getattr(layer, "containing_module_origin", None)
            if module_name in mods or containing == module_name:
                best = i
            # Also check nested containing modules
            nested = getattr(layer, "containing_modules_origin_nested", None)
            if nested and module_name in str(nested):
                best = i
        if best < 0:
            raise ValueError(f"Module '{module_name}' not found in the traced graph.")
        return best

    # ------------------------------------------------------------------
    # 3b. layer profiling & adaptive split-point selection
    # ------------------------------------------------------------------

    def profile_layers(self) -> List[LayerProfile]:
        """Profile **every** traced layer with FLOPs and smashed-data-size.

        Returns a list of ``LayerProfile`` — one per layer — containing
        cumulative FLOPs, intermediate tensor numel, and a privacy-leakage
        heuristic.  This is the universalised version of the previous
        ``config.layer_info`` dictionary.
        """
        self._ensure_traced()
        profiles: List[LayerProfile] = []
        cumulative_flops = 0.0

        # Compute total input numel for privacy normalisation
        input_numel = 1
        for layer in self._layer_list:
            if getattr(layer, "layer_type", "") == "input":
                shape = getattr(layer, "tensor_shape", None)
                if shape:
                    input_numel = int(np.prod(shape))
                break

        for i, layer in enumerate(self._layer_list):
            flops = _estimate_layer_flops(layer)
            cumulative_flops += flops

            shape = getattr(layer, "tensor_shape", None)
            smashed = int(np.prod(shape)) if shape else 0

            privacy = min(float(smashed) / float(max(input_numel, 1)), 1.0)

            profiles.append(LayerProfile(
                index=i,
                label=getattr(layer, "layer_label", f"layer_{i}"),
                cumulative_flops=cumulative_flops,
                smashed_data_size=smashed,
                output_shape=shape,
                privacy_leakage=privacy,
            ))

        return profiles

    def get_candidate_split_points(
        self,
        *,
        only_parametric: bool = False,
        skip_input_output: bool = True,
        min_index: int = 1,
        max_index: int | None = None,
    ) -> List[LayerProfile]:
        """Return a filtered set of *candidate* split points.

        By default all layers (except the first input and last output) are
        candidates.  Set ``only_parametric=True`` to restrict to layers
        that carry parameters (Conv, Linear, BN, etc.) — this mirrors
        the previous ``actionList`` which only contains meaningful split-point layer
        indices (e.g. ``[1,4,7,9,10,13]`` for VGG9).

        Parameters
        ----------
        only_parametric : bool
            If ``True``, only include layers where ``has_params`` is ``True``.
        skip_input_output : bool
            Filter out pure input / output / buffer layers.
        min_index : int
            Minimum layer index (avoid splitting before any computation).
        max_index : int | None
            Maximum layer index (``None`` → second-to-last layer).

        Returns
        -------
        list[LayerProfile] — filtered candidates.
        """
        all_profiles = self.profile_layers()

        if max_index is None:
            max_index = len(all_profiles) - 2  # don't split at the very last layer

        candidates: List[LayerProfile] = []
        for p in all_profiles:
            if p.index < min_index or p.index > max_index:
                continue
            layer = self._layer_list[p.index]
            lt = getattr(layer, "layer_type", "")
            if skip_input_output and lt in ("input", "output", "buffer"):
                continue
            has_params = bool(
                (getattr(layer, "num_param_tensors", 0) or 0) > 0
                or (getattr(layer, "num_params_total", 0) or 0) > 0
            )
            if only_parametric and not has_params:
                continue
            candidates.append(p)

        if not candidates:
            logger.warning(
                "[UniversalSplit] No candidates after filtering — "
                "returning all non-input/output layers."
            )
            candidates = [
                p for p in all_profiles
                if getattr(self._layer_list[p.index], "layer_type", "")
                   not in ("input", "output", "buffer")
            ]

        return candidates

    def create_split_selector(self, *, only_parametric: bool = False) -> SplitPointSelector:
        """Create a split-point selector from current candidate profiles."""
        candidates = self.get_candidate_split_points(only_parametric=only_parametric)
        return SplitPointSelector(candidates)


    # ------------------------------------------------------------------
    # 4. Edge-side forward (head partition)
    # ------------------------------------------------------------------

    def edge_forward(
        self,
        x: torch.Tensor,
        *,
        device: torch.device | str | None = None,
        return_replay_state: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        """Run the **head** partition (edge side) and return the intermediate
        activations at the split point.

        Parameters
        ----------
        x : Tensor
            Model input.
        device : optional
            Device override.

        Returns
        -------
        Tensor | dict — by default returns only the split-point intermediate
        tensor. When ``return_replay_state=True``, returns a dict payload that
        also includes all cross-boundary parent activations required for
        faithful cloud-side replay in a separate process.
        """
        self._ensure_split()
        dev = torch.device(device) if device else self.device

        if dev:
            x = _move_to_device(x, dev)

        if self._use_proxy_split:
            with torch.no_grad():
                mh = tl.log_forward_pass(
                    self._model,
                    x,
                    layers_to_save="all",
                    save_function_args=False,
                )
            self._proxy_input = _detach_clone_any(x)
            act = mh.layer_list[self._split_index].activation
            t = _extract_first_tensor(act)
            if t is None:
                t = _extract_first_tensor(x)
            if t is None:
                raise RuntimeError("Proxy split could not extract tensor activation at split point.")
            inter = t.detach().clone()
            if return_replay_state:
                return {
                    "intermediate": inter,
                    "boundary_activations": {},
                    "split_index": self._split_index,
                    "split_label": getattr(self._layer_list[self._split_index], "layer_label", ""),
                }
            return inter

        _set_training_flag(self._layer_list, training=False)
        with torch.no_grad():
            _, intermediate = _partition_forward_head(
                self._layer_list,
                self._label2idx,
                x,
                self._split_index,
                model=self._model,
            )

        if not return_replay_state:
            return intermediate

        split_label = getattr(self._layer_list[self._split_index], "layer_label", "")
        needed = _collect_tail_boundary_labels(
            self._layer_list,
            self._label2idx,
            self._split_index,
        )
        boundary_activations: dict[str, Any] = {}
        for label in needed:
            if label == split_label:
                continue
            idx = self._label2idx.get(label)
            if idx is None:
                continue
            act = _layer_get_activation(self._layer_list[idx])
            if act is not None:
                boundary_activations[label] = _detach_clone_any(act)

        return {
            "intermediate": intermediate,
            "boundary_activations": boundary_activations,
            "split_index": self._split_index,
            "split_label": split_label,
        }

    # ------------------------------------------------------------------
    # 5. Cloud-side forward (tail partition) — inference
    # ------------------------------------------------------------------

    def cloud_forward(
        self,
        intermediate: torch.Tensor | dict[str, Any],
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Run the **tail** partition (cloud side) for inference.

        Parameters
        ----------
        intermediate : Tensor
            Activations produced by ``edge_forward``.
        device : optional
            Device override.

        Returns
        -------
        Tensor — final model output.
        """
        self._ensure_split()
        dev = torch.device(device) if device else self.device

        boundary_activations: dict[str, Any] = {}
        if isinstance(intermediate, dict):
            payload = intermediate
            if "intermediate" not in payload:
                raise ValueError("Replay payload dict must contain key 'intermediate'.")

            payload_split_idx = payload.get("split_index")
            if payload_split_idx is not None and int(payload_split_idx) != int(self._split_index):
                raise ValueError(
                    f"Replay payload split_index={payload_split_idx} does not match current split_index={self._split_index}."
                )

            intermediate = payload["intermediate"]
            boundary_activations = payload.get("boundary_activations", {}) or {}

            required = _collect_tail_boundary_labels(
                self._layer_list,
                self._label2idx,
                self._split_index,
            )
            split_label = getattr(self._layer_list[self._split_index], "layer_label", "")
            missing = sorted([lb for lb in required if lb != split_label and lb not in boundary_activations])
            if missing:
                raise RuntimeError(
                    "Replay payload is missing required boundary activations: "
                    + ", ".join(missing[:8])
                    + (" ..." if len(missing) > 8 else "")
                )

        if dev and isinstance(intermediate, torch.Tensor) and intermediate.device != dev:
            intermediate = intermediate.to(dev)
        if dev and boundary_activations:
            boundary_activations = _move_to_device(boundary_activations, dev)

        if self._use_proxy_split:
            if self._proxy_input is None:
                raise RuntimeError("Proxy split mode requires edge_forward to run before cloud_forward.")
            x = self._proxy_input
            if dev:
                x = _move_to_device(x, dev)
            with torch.no_grad():
                if isinstance(x, tuple):
                    return self._model(*x)
                return self._model(x)

        _set_training_flag(self._layer_list, training=False)
        for label, value in boundary_activations.items():
            idx = self._label2idx.get(label)
            if idx is not None:
                _layer_set_activation(self._layer_list[idx], value)

        with torch.no_grad():
            output = _partition_forward_tail(
                self._layer_list,
                self._label2idx,
                intermediate,
                self._split_index,
                model=self._model,
            )
        return output

    # ------------------------------------------------------------------
    # 6. Cloud-side training step (tail partition)
    # ------------------------------------------------------------------

    def cloud_train_step(
        self,
        intermediate: torch.Tensor,
        targets: Any = None,
        loss_fn: Callable | None = None,
        *,
        device: torch.device | str | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward + loss computation on the cloud tail.

        Parameters
        ----------
        intermediate : Tensor
            Activations from the edge.  ``requires_grad`` will be enabled so
            that gradients can propagate through the split boundary if needed.
        targets : Any
            Ground-truth targets for loss computation.
        loss_fn : Callable
            ``loss_fn(output, targets) -> scalar loss``.  If ``None`` the user
            is expected to compute the loss externally from the returned
            output.
        device : device override.

        Returns
        -------
        (output, loss)  where *loss* is ``torch.tensor(0.)`` when no
        ``loss_fn`` is provided.
        """
        self._ensure_split()
        dev = torch.device(device) if device else self.device

        if dev and intermediate.device != dev:
            intermediate = intermediate.to(dev)

        # Enable grad on the intermediate so backprop can cross the boundary
        intermediate = intermediate.detach().requires_grad_(True)

        if self._use_proxy_split:
            if self._proxy_input is None:
                raise RuntimeError("Proxy split mode requires edge_forward to run before cloud_train_step.")
            x = self._proxy_input
            if dev:
                x = _move_to_device(x, dev)

            if dev and targets is not None:
                targets = _move_to_device(targets, dev)

            # Native detection-model training path (returns a loss dict).
            if (
                hasattr(self._model, "rpn")
                and hasattr(self._model, "roi_heads")
                and targets is not None
                and loss_fn is None
            ):
                self._model.train()
                images = x[0] if isinstance(x, tuple) and len(x) == 1 else x
                if isinstance(images, torch.Tensor):
                    if images.dim() == 4:
                        images = [img for img in images]
                    else:
                        images = [images]
                target_list = targets if isinstance(targets, list) else [targets]
                loss_dict = self._model(images, target_list)
                if isinstance(loss_dict, dict) and loss_dict:
                    total_loss = sum(loss_dict.values())
                    return loss_dict, total_loss

            # For non-detection proxy models (e.g. ultralytics), keep eval
            # mode to preserve inference-like output structure used by tests.
            self._model.eval()
            if isinstance(x, tuple):
                output = self._model(*x)
            else:
                output = self._model(x)
            if loss_fn is not None and targets is not None:
                loss = loss_fn(output, targets)
            else:
                loss = torch.tensor(0.0, device=dev)
            return output, loss

        _set_training_flag(self._layer_list, training=True)

        output = _partition_forward_tail_train(
            self._layer_list,
            self._label2idx,
            intermediate,
            self._split_index,
            model=self._model,
        )

        if loss_fn is not None and targets is not None:
            if dev and hasattr(targets, "to"):
                targets = targets.to(dev)
            loss = loss_fn(output, targets)
        else:
            loss = torch.tensor(0.0, device=dev)

        return output, loss

    # ------------------------------------------------------------------
    # 7. Parameter management
    # ------------------------------------------------------------------

    def freeze_head(self) -> None:
        """Freeze all parameters in the head (edge) partition."""
        self._ensure_split()
        _freeze_head_params(self._layer_list, self._split_index, self._model)

    def unfreeze_tail(self) -> None:
        """Unfreeze all parameters in the tail (cloud) partition."""
        self._ensure_split()
        _unfreeze_tail_params(self._layer_list, self._split_index, self._model)

    def get_tail_trainable_params(self) -> list[torch.Tensor]:
        """Return a list of trainable parameters in the cloud tail."""
        self._ensure_split()
        return _get_trainable_params(self._layer_list, self._split_index, self._model)

    def get_all_params(self) -> list[torch.Tensor]:
        """Return all parameters of the full model."""
        self._ensure_traced()
        return _get_all_params(self._layer_list, self._model)

    def load_weights(self, weights: list[torch.Tensor]) -> None:
        self._ensure_traced()
        _load_weights_into_layers(self._layer_list, self._model, weights)

    # ------------------------------------------------------------------
    # 8. Serialisation for network transmission
    # ------------------------------------------------------------------

    @staticmethod
    def serialise_intermediate(
        tensor: Any,
        *,
        compress: bool = False,
    ) -> bytes:
        """Serialise an intermediate tensor for network transfer.

        Parameters
        ----------
        tensor : Any
        compress : bool
            If ``True``, apply zlib compression.

        Returns
        -------
        bytes — serialised payload.
        """
        buf = io.BytesIO()
        if isinstance(tensor, torch.Tensor):
            payload = tensor.detach().cpu()
        else:
            payload = _move_to_device(_detach_clone_any(tensor), torch.device("cpu"))
        torch.save(payload, buf)
        data = buf.getvalue()
        if compress:
            import zlib
            data = zlib.compress(data)
        return data

    @staticmethod
    def deserialise_intermediate(
        data: bytes,
        *,
        device: torch.device | str = "cpu",
        compressed: bool = False,
    ) -> Any:
        """Deserialise an intermediate tensor received over the network.

        Parameters
        ----------
        data   : bytes
        device : target device
        compressed : whether zlib decompression is needed

        Returns
        -------
        Deserialised payload on *device*.
        """
        if compressed:
            import zlib
            data = zlib.decompress(data)
        buf = io.BytesIO(data)
        payload = torch.load(buf, map_location=device)
        return payload

    # ------------------------------------------------------------------
    # 9. Full inference (unsplit) via replay
    # ------------------------------------------------------------------

    def full_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the full model layer-by-layer (no split).  Useful for
        validating that the traced graph is faithful.
        """
        self._ensure_traced()
        if self._use_proxy_split:
            with torch.no_grad():
                return self._model(x)
        _set_training_flag(self._layer_list, training=False)
        with torch.no_grad():
            return _layer_forward_full(self._layer_list, self._label2idx, x, model=self._model)

    # ------------------------------------------------------------------
    # 10. Split-learning training loop (convenience wrapper)
    # ------------------------------------------------------------------

    def split_retrain(
        self,
        train_loader,
        loss_fn: Callable,
        *,
        num_epochs: int = 2,
        lr: float = 0.005,
        device: torch.device | str | None = None,
    ) -> list[float]:
        """Convenience: run a full split-learning training loop.

        Parameters
        ----------
        train_loader : iterable
            Yields ``(input_tensor, target)`` pairs.  The *input_tensor* will
            be forwarded through the head partition first.
        loss_fn : Callable
            ``loss_fn(output, target) -> loss``.
        num_epochs : int
        lr : float
        device : device override

        Returns
        -------
        list[float] — per-epoch average loss.
        """
        self._ensure_split()
        dev = torch.device(device) if device else self.device

        # Freeze head, train tail
        self.freeze_head()
        self.unfreeze_tail()

        tail_params = self.get_tail_trainable_params()
        if not tail_params:
            logger.warning("[UniversalSplit] No trainable params in the tail partition!")
            return []

        optimizer = torch.optim.SGD(tail_params, lr=lr, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        epoch_losses: list[float] = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n = 0

            for inputs, targets in train_loader:
                if dev:
                    if hasattr(inputs, "to"):
                        inputs = inputs.to(dev)
                    if hasattr(targets, "to"):
                        targets = targets.to(dev)

                # Edge forward (no grad)
                intermediate = self.edge_forward(inputs)

                # Cloud forward + loss (with grad)
                output, loss = self.cloud_train_step(
                    intermediate, targets, loss_fn, device=dev,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n += 1

            lr_scheduler.step()
            avg = epoch_loss / max(n, 1)
            epoch_losses.append(avg)
            logger.info(
                "[UniversalSplit] Epoch {}/{} — samples={}, avg_loss={:.4f}",
                epoch + 1, num_epochs, n, avg,
            )

        return epoch_losses

    # ------------------------------------------------------------------
    # State-dict helpers (for transmitting trained weights)
    # ------------------------------------------------------------------

    def get_tail_state_dict(self) -> dict[str, torch.Tensor]:
        """Build a pseudo state-dict from the tail partition's parameters.

        Returns a dict mapping ``"layer_{idx}_param_{j}"`` → Tensor for each
        parameter in the cloud partition.  This can be serialised and sent
        back to the edge.
        """
        self._ensure_split()
        sd: dict[str, torch.Tensor] = {}
        for i in range(self._split_index + 1, len(self._layer_list)):
            layer = self._layer_list[i]
            params = _get_param_tensors_for_layer(layer, self._model)
            for j, p in enumerate(params):
                sd[f"layer_{i}_param_{j}"] = p.detach().cpu()
        return sd

    def load_tail_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        """Load a pseudo state-dict produced by ``get_tail_state_dict``."""
        self._ensure_split()
        for key, val in sd.items():
            parts = key.split("_")
            # "layer_{i}_param_{j}"
            layer_idx = int(parts[1])
            param_idx = int(parts[3])
            layer = self._layer_list[layer_idx]
            params = _get_param_tensors_for_layer(layer, self._model)
            if params and param_idx < len(params):
                params[param_idx].data.copy_(val)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_traced(self) -> None:
        if not self._traced:
            raise RuntimeError(
                "Model has not been traced yet.  Call splitter.trace(model, sample_input) first."
            )

    def _ensure_split(self) -> None:
        self._ensure_traced()
        if self._split_index is None:
            raise RuntimeError(
                "Split point not set.  Call splitter.split(layer_index=...) first."
            )

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if not self._traced:
            return "UniversalModelSplitter(not traced)"
        name = self._model_history.model_name if self._model_history else "?"
        n = len(self._layer_list)
        sp = self._split_index
        if sp is not None:
            return f"UniversalModelSplitter({name}, {n} layers, split@{sp})"
        return f"UniversalModelSplitter({name}, {n} layers, not split)"


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: feature-cache based training (integration with existing
# pipeline — replaces model_split.split_retrain for the universal case)
# ═══════════════════════════════════════════════════════════════════════════


def _universal_split_retrain_das(
    model: nn.Module,
    cache_path: str,
    all_indices: List[int],
    gt_annotations: Dict[int, dict],
    device: torch.device,
    num_epoch: int = 2,
    lr: float = 0.005,
    das_bn_only: bool = False,
    das_probe_samples: int = 10,
) -> None:
    """DAS-enabled training path for detection models (rpn + roi_heads).

    This bypasses TorchLens replay and uses native module forward so that
    the AutoFreeze custom autograd functions are invoked, enabling dynamic
    activation sparsity during backpropagation.

    Requires the model to have ``backbone``, ``rpn``, ``roi_heads``
    attributes (torchvision detection model convention).
    """
    from model_management.model_split import (
        load_feature_cache,
        server_side_train_step,
        transform_targets_to_feature_space,
        build_image_list,
    )

    # Freeze backbone, unfreeze rpn + roi_heads
    for p in model.parameters():
        p.requires_grad = False
    for p in model.rpn.parameters():
        p.requires_grad = True
    for p in model.roi_heads.parameters():
        p.requires_grad = True

    # Apply DAS to tail modules
    das_trainer = apply_das_to_tail(
        model, ["rpn", "roi_heads"],
        bn_only=das_bn_only, device=device,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        logger.warning("[UnivSplitRetrain/DAS] No trainable params in tail!")
        return

    optimizer = torch.optim.SGD(trainable, lr=lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    gt_set = set(gt_annotations.keys())
    model.to(device)

    for epoch in range(num_epoch):
        epoch_loss = 0.0
        n_samples = 0
        indices = all_indices.copy()
        random.shuffle(indices)

        # ---- DAS Phase 1: probe gradient importance ----
        _das_probe_for_universal(
            das_trainer, model, cache_path, indices,
            gt_annotations, device, das_probe_samples,
        )
        das_trainer.activate_sparsity()

        # ---- DAS Phase 2: train with activation sparsity ----
        model.train()
        model.backbone.eval()  # BN frozen in backbone

        for idx in indices:
            try:
                data = load_feature_cache(cache_path, idx)
            except (FileNotFoundError, OSError, RuntimeError, KeyError, ValueError, pickle.UnpicklingError):
                continue

            features = data["features"]
            image_sizes = data["image_sizes"]
            tensor_shape = data["tensor_shape"]
            original_sizes = data["original_sizes"]

            if idx in gt_set:
                gt = gt_annotations[idx]
                boxes, labels = gt["boxes"], gt["labels"]
            else:
                boxes = data.get("pseudo_boxes", [])
                labels = data.get("pseudo_labels", [])

            if not boxes or not labels:
                continue

            targets = [{
                "boxes":  torch.tensor(boxes,  dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
            }]
            targets = transform_targets_to_feature_space(
                targets, original_sizes, image_sizes,
            )

            try:
                loss_dict = server_side_train_step(
                    model, features, image_sizes, tensor_shape, targets, device,
                )
                total_loss = sum(loss_dict.values())
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()
                n_samples += 1
            except Exception as exc:
                logger.warning("[UnivSplitRetrain/DAS] Error on frame {}: {}", idx, exc)

        lr_scheduler.step()
        avg = epoch_loss / max(n_samples, 1)
        logger.info(
            "[UnivSplitRetrain/DAS] Epoch {}/{} — samples={}, avg_loss={:.4f}",
            epoch + 1, num_epoch, n_samples, avg,
        )

    das_trainer.deactivate_sparsity()
    stats = das_trainer.get_memory_stats()
    logger.info(
        "[UnivSplitRetrain/DAS] Training done. Activation compression: {:.1%}",
        stats.get("compression_ratio", 1.0),
    )


def _das_probe_for_universal(
    das_trainer,
    model: nn.Module,
    cache_path: str,
    indices: List[int],
    gt_annotations: Dict[int, dict],
    device: torch.device,
    probe_samples: int = 10,
) -> None:
    """Probe gradient importance for DAS in the universal training path."""
    from model_management.model_split import (
        load_feature_cache,
        server_side_train_step,
        transform_targets_to_feature_space,
    )
    gt_set = set(gt_annotations.keys())

    for idx in indices[:probe_samples]:
        try:
            data = load_feature_cache(cache_path, idx)
        except (FileNotFoundError, OSError, RuntimeError, KeyError, ValueError, pickle.UnpicklingError):
            continue

        features = data["features"]
        image_sizes = data["image_sizes"]
        tensor_shape = data["tensor_shape"]
        original_sizes = data["original_sizes"]

        if idx in gt_set:
            gt = gt_annotations[idx]
            boxes, labels = gt["boxes"], gt["labels"]
        else:
            boxes = data.get("pseudo_boxes", [])
            labels = data.get("pseudo_labels", [])
        if not boxes or not labels:
            continue

        targets = [{
            "boxes":  torch.tensor(boxes,  dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }]
        targets = transform_targets_to_feature_space(targets, original_sizes, image_sizes)

        def _probe_fn(
            _m=model, _f=features, _is=image_sizes,
            _ts=tensor_shape, _t=targets, _d=device,
        ):
            return server_side_train_step(_m, _f, _is, _ts, _t, _d)

        try:
            das_trainer.probe_with_targets(_probe_fn)
            return  # one probe is sufficient
        except Exception as exc:
            logger.debug("[DAS] Probe failed on frame {}: {}", idx, exc)

    logger.debug("[DAS] Could not probe any sample; using zero pruning.")


def universal_split_retrain(
    model: nn.Module,
    sample_input: torch.Tensor,
    split_layer: int | str,
    cache_path: str,
    all_indices: List[int],
    gt_annotations: Dict[int, dict],
    device: torch.device,
    num_epoch: int = 2,
    lr: float = 0.005,
    loss_fn: Callable | None = None,
    das_enabled: bool = False,
    das_bn_only: bool = False,
    das_probe_samples: int = 10,
) -> None:
    """High-level entry point that mirrors ``model_split.split_retrain`` but
    works for any model and any split point.

    Parameters
    ----------
    model : nn.Module
        Full model.
    sample_input : Tensor
        Representative input for tracing.
    split_layer : int | str
        Layer index or label at which to split.
    cache_path : str
        Directory containing ``features/<idx>.pt`` files.
    all_indices : list[int]
        Frame indices to use.
    gt_annotations : dict
        ``{frame_idx: {"boxes": …, "labels": …}}`` for drift frames.
    device : torch.device
    num_epoch : int
    lr : float
    loss_fn : Callable | None
        Custom loss function.  If ``None`` a default cross-entropy is used.
    das_enabled : bool
        Enable SURGEON-style Dynamic Activation Sparsity for memory-
        efficient training.  When enabled *and* the model is a detection
        model (has ``rpn`` + ``roi_heads``), DAS is applied to the tail
        modules and training uses the native module forward path for full
        DAS benefit.  Otherwise, standard TorchLens replay is used (DAS
        has no effect on the replay path).
    das_bn_only : bool
        When DAS is active, only update BN parameters.
    das_probe_samples : int
        Number of samples for gradient-importance probing.
    """
    # ---- DAS path for detection models (rpn + roi_heads) ----
    if das_enabled and hasattr(model, "rpn") and hasattr(model, "roi_heads"):
        _universal_split_retrain_das(
            model=model, cache_path=cache_path,
            all_indices=all_indices, gt_annotations=gt_annotations,
            device=device, num_epoch=num_epoch, lr=lr,
            das_bn_only=das_bn_only, das_probe_samples=das_probe_samples,
        )
        return

    if das_enabled:
        logger.warning(
            "[UniversalSplitRetrain] DAS requested but model is not a "
            "detection model (rpn/roi_heads not found). Falling back to "
            "standard TorchLens replay (DAS has no effect).",
        )

    splitter = UniversalModelSplitter(device=device)
    splitter.trace(model, sample_input)

    if isinstance(split_layer, str):
        splitter.split(layer_label=split_layer)
    else:
        splitter.split(layer_index=split_layer)

    splitter.freeze_head()
    splitter.unfreeze_tail()

    tail_params = splitter.get_tail_trainable_params()
    if not tail_params:
        logger.warning("[UniversalSplitRetrain] No trainable params in the tail!")
        return

    optimizer = torch.optim.SGD(tail_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    gt_set = set(gt_annotations.keys())

    for epoch in range(num_epoch):
        epoch_loss = 0.0
        n_samples = 0
        indices = all_indices.copy()
        random.shuffle(indices)

        for idx in indices:
            feat_path = os.path.join(cache_path, "features", f"{idx}.pt")
            if not os.path.exists(feat_path):
                continue

            try:
                data = torch.load(feat_path, map_location="cpu")
            except (OSError, RuntimeError, pickle.UnpicklingError, EOFError, ValueError) as exc:
                logger.warning("[UniversalSplitRetrain] Cannot load {}: {}", feat_path, exc)
                continue

            intermediate = data.get("intermediate")
            if intermediate is None:
                # Backward-compat: try to load from old-style feature cache
                features = data.get("features")
                if features is not None:
                    # Use the first feature map as the intermediate
                    if isinstance(features, dict):
                        intermediate = list(features.values())[0]
                    elif isinstance(features, OrderedDict):
                        intermediate = next(iter(features.values()))
                    else:
                        intermediate = features

            if intermediate is None:
                continue

            # Build targets
            if idx in gt_set:
                gt = gt_annotations[idx]
                boxes = gt["boxes"]
                labels = gt["labels"]
            else:
                boxes = data.get("pseudo_boxes", [])
                labels = data.get("pseudo_labels", [])

            if not boxes or not labels:
                continue

            targets_t = {
                "boxes": torch.tensor(boxes, dtype=torch.float32).to(device),
                "labels": torch.tensor(labels, dtype=torch.int64).to(device),
            }

            intermediate = intermediate.to(device).detach().requires_grad_(True)

            try:
                output, loss = splitter.cloud_train_step(
                    intermediate,
                    targets_t,
                    loss_fn,
                    device=device,
                )

                if loss_fn is None:
                    # If no loss_fn provided, compute a default: sum of output
                    loss = output.sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_samples += 1
            except Exception as exc:
                logger.warning("[UniversalSplitRetrain] Error on frame {}: {}", idx, exc)
                continue

        lr_scheduler.step()
        avg = epoch_loss / max(n_samples, 1)
        logger.info(
            "[UniversalSplitRetrain] Epoch {}/{} — samples={}, avg_loss={:.4f}",
            epoch + 1, num_epoch, n_samples, avg,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: extract & cache intermediate features at the split point
# (replaces model_split.extract_backbone_features for arbitrary models)
# ═══════════════════════════════════════════════════════════════════════════

def extract_split_features(
    splitter: UniversalModelSplitter,
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    """Run the edge partition and return intermediate activation on CPU.

    This is the universal replacement for ``extract_backbone_features``.
    """
    return splitter.edge_forward(input_tensor).cpu()


def save_split_feature_cache(
    cache_path: str,
    frame_index: int,
    intermediate: torch.Tensor,
    is_drift: bool,
    pseudo_boxes: list | None = None,
    pseudo_labels: list | None = None,
    pseudo_scores: list | None = None,
    extra_metadata: dict | None = None,
) -> str:
    """Persist one frame's intermediate features to ``<cache>/features/<idx>.pt``.

    This replaces ``model_split.save_feature_cache`` with a simpler interface
    that does not assume Faster R-CNN's ``ImageList`` structure.
    """
    feat_dir = os.path.join(cache_path, "features")
    os.makedirs(feat_dir, exist_ok=True)
    out_path = os.path.join(feat_dir, f"{frame_index}.pt")
    payload = {
        "intermediate": intermediate,
        "is_drift": is_drift,
        "pseudo_boxes": pseudo_boxes or [],
        "pseudo_labels": pseudo_labels or [],
        "pseudo_scores": pseudo_scores or [],
    }
    if extra_metadata:
        payload.update(extra_metadata)
    torch.save(payload, out_path)
    return out_path


def load_split_feature_cache(cache_path: str, frame_index: int) -> dict:
    """Load one frame's cached intermediate features."""
    path = os.path.join(cache_path, "features", f"{frame_index}.pt")
    # Cached payload may include non-tensor metadata; allow full load for
    # trusted artifacts generated by this project.
    return torch.load(path, map_location="cpu", weights_only=False)
