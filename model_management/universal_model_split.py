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

Split-point selection is inspired by:

* **HSFL** (ICWS 2023) — ``SASA-cloud/ICWS-23-HSFL``.  HSFL profiles each
  candidate split layer with ``(cumulative_FLOPs, smashed_data_size)`` and
  uses a **LinUCB contextual bandit** to adaptively choose the optimal split
  point at each federated round, balancing *training latency* and *privacy
  leakage* (proportional to intermediate tensor dimensionality).  We
  generalise their approach to arbitrary models: ``profile_layers()``
  automatically computes FLOPs / tensor size / privacy-leakage estimates,
  ``SplitPointSelector`` wraps the LinUCB agent, and ``select_split_point()``
  on ``UniversalModelSplitter`` provides a one-call API.

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
# TorchLens is an *optional* dependency.  When it is available we use it
# for the model-agnostic tracing.  When it is not installed, the class falls
# back to the legacy Faster R-CNN specific helpers (kept in model_split.py).
# ---------------------------------------------------------------------------
try:
    import torchlens as tl
    from torchlens.tensor_log import TensorLogEntry

    _HAS_TORCHLENS = True
except ImportError:
    _HAS_TORCHLENS = False
    TensorLogEntry = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Dynamic Activation Sparsity (SURGEON-style layer pruning).
# ---------------------------------------------------------------------------
try:
    from model_management.activation_sparsity import (
        DASTrainer,
        apply_das_to_tail,
    )
    _HAS_DAS = True
except ImportError:
    _HAS_DAS = False


# ═══════════════════════════════════════════════════════════════════════════
# Helper: build label→index map and prepare layers for replay
# ═══════════════════════════════════════════════════════════════════════════

def _prepare_replay_graph(layer_list: list) -> dict[str, int]:
    """Build a ``{layer_label: index}`` mapping and call ``prepare_replay``
    on every non-trivial layer so that ``replay_fast`` can be used later.

    This mirrors the ``prepare_replay_graph`` function from Shawarma /
    torchlens test_replay.py.
    """
    label2idx: dict[str, int] = {}
    for i, layer in enumerate(layer_list):
        label2idx[layer.layer_label] = i
        if getattr(layer, "func_applied_name", None) not in (None, "none"):
            try:
                layer.prepare_replay()
            except Exception:
                pass  # some layers (input/output/buffer) don't need prep
    return label2idx


# ═══════════════════════════════════════════════════════════════════════════
# Helper: layer-by-layer forward pass (full model, eval mode)
# ═══════════════════════════════════════════════════════════════════════════

def _layer_forward_full(
    layer_list: list,
    label2idx: dict[str, int],
    x: torch.Tensor,
) -> torch.Tensor:
    """Execute the full model layer-by-layer using the TorchLens replay
    mechanism.  This is the Shawarma ``forward_eval`` equivalent.

    Parameters
    ----------
    layer_list : list[TensorLogEntry]
    label2idx  : dict mapping layer_label → list index
    x          : input tensor

    Returns
    -------
    torch.Tensor — model output
    """
    for layer in layer_list:
        func_name = getattr(layer, "func_applied_name", None)
        layer_type = getattr(layer, "layer_type", None)

        # No-op layers: input / output / buffer
        if func_name == "none":
            if layer_type == "input":
                layer.tensor_contents = x
            elif layer_type == "output":
                parent_idx = label2idx[layer.parent_layers[0]]
                layer.tensor_contents = layer_list[parent_idx].tensor_contents
            # buffer layers keep their pre-existing tensor_contents
            continue

        # Gather inputs from parent layers
        x_in: list = []
        buffer_in: list = []
        op_num = layer.operation_num

        for plabel in layer.parent_layers:
            p = layer_list[label2idx[plabel]]
            if p.layer_type == "buffer":
                buffer_in.append(p.tensor_contents)
            else:
                tc = p.tensor_contents
                if tc is None and p.operation_num == op_num - 1:
                    x_in.append(x)
                else:
                    x_in.append(tc)
                # Memory: release parent once its last child has consumed it
                if (
                    p.layer_type not in ("input", "buffer")
                    and p.child_layers
                    and label2idx.get(p.child_layers[-1], 0) <= label2idx[layer.layer_label]
                ):
                    p.tensor_contents = None

        # Execute
        x = layer.replay_fast(x_in, buffer_in)

        # Memory: aggressively free intermediates
        if (
            layer.child_layers
            and layer_list[label2idx[layer.child_layers[-1]]].operation_num <= op_num + 1
        ):
            layer.tensor_contents = None
        else:
            layer.tensor_contents = x

    return x


# ═══════════════════════════════════════════════════════════════════════════
# Helper: partition forward (head / tail)
# ═══════════════════════════════════════════════════════════════════════════

def _partition_forward_head(
    layer_list: list,
    label2idx: dict[str, int],
    x: torch.Tensor,
    split_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run layers ``[0 .. split_index]`` and return ``(final_output, intermediate)``.

    *intermediate* is the activation at ``split_index`` — this is what the
    edge sends to the cloud.
    """
    intermediate: torch.Tensor | None = None

    for idx, layer in enumerate(layer_list):
        if idx > split_index:
            break

        func_name = getattr(layer, "func_applied_name", None)
        layer_type = getattr(layer, "layer_type", None)

        if func_name == "none":
            if layer_type == "input":
                layer.tensor_contents = x
            elif layer_type == "buffer":
                pass  # keep existing contents
            elif layer_type == "output":
                if layer.parent_layers:
                    parent_idx = label2idx[layer.parent_layers[0]]
                    layer.tensor_contents = layer_list[parent_idx].tensor_contents
            if idx == split_index:
                intermediate = (
                    layer.tensor_contents.detach().clone()
                    if layer.tensor_contents is not None
                    else x.detach().clone()
                )
            continue

        # Gather inputs
        x_in: list = []
        buffer_in: list = []
        op_num = layer.operation_num

        for plabel in layer.parent_layers:
            p = layer_list[label2idx[plabel]]
            if p.layer_type == "buffer":
                buffer_in.append(p.tensor_contents)
            else:
                tc = p.tensor_contents
                if tc is None and p.operation_num == op_num - 1:
                    x_in.append(x)
                else:
                    x_in.append(tc)
                # Cleanup
                if (
                    p.layer_type not in ("input", "buffer")
                    and p.child_layers
                    and label2idx.get(p.child_layers[-1], 0) <= label2idx[layer.layer_label]
                ):
                    p.tensor_contents = None

        x = layer.replay_fast(x_in, buffer_in)
        layer.tensor_contents = x

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
) -> torch.Tensor:
    """Run layers ``[split_index+1 .. end]`` given the intermediate tensor
    produced by the head.

    Missing parents (i.e. layers before split_index that are not in the
    current partition) are replaced by ``x_inter`` the first time they are
    needed.  This mirrors the Shawarma ``partition_forward_eval`` pattern.
    """
    first_in = True
    x = x_inter

    for idx, layer in enumerate(layer_list):
        if idx <= split_index:
            continue

        func_name = getattr(layer, "func_applied_name", None)
        layer_type = getattr(layer, "layer_type", None)

        if func_name == "none":
            if layer_type == "buffer":
                pass  # keep buffer contents
            elif layer_type == "output":
                if layer.parent_layers:
                    parent_label = layer.parent_layers[0]
                    if parent_label in label2idx and label2idx[parent_label] > split_index:
                        layer.tensor_contents = layer_list[label2idx[parent_label]].tensor_contents
                    else:
                        layer.tensor_contents = x
            else:
                layer.tensor_contents = None
            continue

        x_in: list = []
        buffer_in: list = []

        for plabel in layer.parent_layers:
            if plabel not in label2idx or label2idx[plabel] <= split_index:
                # Parent is in the head partition — inject intermediate
                if first_in:
                    x_in.append(x_inter)
                    first_in = False
                continue

            p = layer_list[label2idx[plabel]]
            if p.layer_type == "buffer":
                buffer_in.append(p.tensor_contents)
            else:
                tc = p.tensor_contents
                if tc is not None:
                    x_in.append(tc)
                    # Cleanup
                    if (
                        p.child_layers
                        and p.child_layers[-1] in label2idx
                        and label2idx[p.child_layers[-1]] <= label2idx[layer.layer_label]
                    ):
                        p.tensor_contents = None
                elif p.operation_num == layer.operation_num - 1:
                    x_in.append(x)

        x = layer.replay_fast(x_in, buffer_in)

        # Cleanup current layer
        if (
            layer.child_layers
            and layer.child_layers[-1] in label2idx
            and layer_list[label2idx[layer.child_layers[-1]]].operation_num <= layer.operation_num + 1
        ):
            layer.tensor_contents = None
        else:
            layer.tensor_contents = x

    return x


def _partition_forward_tail_train(
    layer_list: list,
    label2idx: dict[str, int],
    x_inter: torch.Tensor,
    split_index: int,
) -> torch.Tensor:
    """Training-mode forward through the tail partition.

    Unlike the eval variant this does **not** aggressively release
    ``tensor_contents`` so that the autograd graph stays intact for
    ``backward()``.
    """
    first_in = True
    x = x_inter

    for idx, layer in enumerate(layer_list):
        if idx <= split_index:
            continue

        func_name = getattr(layer, "func_applied_name", None)
        layer_type = getattr(layer, "layer_type", None)

        if func_name == "none":
            if layer_type == "input":
                layer.tensor_contents = x_inter
            elif layer_type == "output":
                if layer.parent_layers:
                    plabel = layer.parent_layers[0]
                    if plabel in label2idx and label2idx[plabel] > split_index:
                        layer.tensor_contents = layer_list[label2idx[plabel]].tensor_contents
                    else:
                        layer.tensor_contents = x
            continue

        x_in: list = []
        buffer_in: list = []

        for plabel in layer.parent_layers:
            if plabel not in label2idx or label2idx[plabel] <= split_index:
                if first_in:
                    x_in.append(x_inter)
                    first_in = False
                continue

            p = layer_list[label2idx[plabel]]
            if getattr(p, "layer_type", None) == "buffer":
                buffer_in.append(p.tensor_contents)
            else:
                tc = p.tensor_contents
                if tc is not None:
                    x_in.append(tc)
                else:
                    x_in.append(x)

        x = layer.replay_fast(x_in, buffer_in)
        layer.tensor_contents = x

    return x


# ═══════════════════════════════════════════════════════════════════════════
# Helper: freeze / unfreeze parameters by partition index
# ═══════════════════════════════════════════════════════════════════════════

def _freeze_head_params(layer_list: list, split_index: int) -> None:
    """Freeze all parameters in layers ``[0 .. split_index]``."""
    limit = min(split_index + 1, len(layer_list))
    for i in range(limit):
        for param in getattr(layer_list[i], "parent_params", []) or []:
            param.requires_grad = False


def _unfreeze_tail_params(layer_list: list, split_index: int) -> None:
    """Ensure all parameters in layers ``[split_index+1 .. end]`` are trainable."""
    for i in range(split_index + 1, len(layer_list)):
        for param in getattr(layer_list[i], "parent_params", []) or []:
            param.requires_grad = True


# ═══════════════════════════════════════════════════════════════════════════
# Helper: parameter access helpers
# ═══════════════════════════════════════════════════════════════════════════

def _get_all_params(layer_list: list, device=None) -> list[torch.Tensor]:
    """Flatten and collect all parameters across all layers."""
    params: list[torch.Tensor] = []
    for layer in layer_list:
        lp = getattr(layer, "parent_params", None)
        if lp:
            if device is None:
                params.extend(lp)
            else:
                params.extend(p.to(device) for p in lp)
    return params


def _get_trainable_params(layer_list: list, split_index: int) -> list[torch.Tensor]:
    """Return all trainable (``requires_grad=True``) params in the tail partition."""
    params: list[torch.Tensor] = []
    for i in range(split_index + 1, len(layer_list)):
        for p in getattr(layer_list[i], "parent_params", []) or []:
            if p.requires_grad:
                params.append(p)
    return params


def _load_weights_into_layers(layer_list: list, weights: list[torch.Tensor]) -> None:
    """Load weights sequentially into the layer list (Shawarma ``load_weights``)."""
    weight_iter = iter(weights)
    for layer in layer_list:
        lp = getattr(layer, "parent_params", None)
        if not lp:
            continue
        for j, _param in enumerate(lp):
            try:
                lp[j] = next(weight_iter)
            except StopIteration:
                return


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
            args = layer.func_all_args_non_tensor
            if args:
                for idx, arg in enumerate(args):
                    if isinstance(arg, bool):
                        args[idx] = training
                        break


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
# Layer profiling data (HSFL-inspired, generalised to arbitrary models)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LayerProfile:
    """Per-candidate-split-point cost profile.

    This mirrors the ``layer_info`` dict from HSFL ``config.py``::

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


# ═══════════════════════════════════════════════════════════════════════════
# Split-point selector: LinUCB bandit + helper strategies (HSFL-style)
# ═══════════════════════════════════════════════════════════════════════════

def _estimate_layer_flops(layer) -> float:
    """Heuristic FLOPs for a single traced layer.

    TorchLens exposes ``func_applied_name``, ``tensor_shape``, and parameter
    counts.  We use these to compute a rough FLOP estimate:

    * **Conv2d**: ``c_out * h * w * k² * c_in``
    * **Linear**: ``out_features * in_features``
    * **BatchNorm / Pool / activation**: 1 × numel (cheap)
    * **Others**: 0

    This is intentionally approximate; the *relative* ordering of candidate
    split points is what matters for the LinUCB context.
    """
    fn = getattr(layer, "func_applied_name", "") or ""
    shape = getattr(layer, "tensor_shape", None)
    num_params = getattr(layer, "num_params_total", 0) or 0

    # Attempt to infer from the function name
    fn_lower = fn.lower()

    if "conv" in fn_lower:
        # FLOPs ≈ out_elements * k² * c_in
        # We approximate k²*c_in from num_params (= c_out*c_in*k*k + bias)
        if shape and len(shape) >= 3:
            out_elements = 1
            for d in shape[1:]:   # skip batch dim
                out_elements *= d
            # num_params ≈ c_out*c_in*k*k (+c_out bias)
            out_channels = shape[1] if len(shape) == 4 else shape[-1]
            if out_channels > 0 and num_params > 0:
                # flops ≈ 2 * num_params * spatial_size
                spatial = out_elements // max(out_channels, 1)
                return float(2 * num_params * spatial)
            return float(2 * num_params) if num_params else float(out_elements)
        return float(2 * num_params) if num_params else 0.0

    if "linear" in fn_lower or "addmm" in fn_lower or "mm" in fn_lower:
        # FLOPs ≈ 2 * in_features * out_features
        return float(2 * num_params) if num_params else 0.0

    if any(kw in fn_lower for kw in ("batch_norm", "layer_norm", "group_norm",
                                      "instance_norm")):
        if shape:
            return float(np.prod(shape[1:]))
        return 0.0

    if any(kw in fn_lower for kw in ("pool", "adaptive_avg", "adaptive_max")):
        if shape:
            return float(np.prod(shape[1:]))
        return 0.0

    if any(kw in fn_lower for kw in ("relu", "gelu", "silu", "sigmoid", "tanh",
                                      "dropout", "add", "mul", "cat")):
        if shape:
            return float(np.prod(shape[1:]))
        return 0.0

    return 0.0


def _estimate_privacy_leakage(
    smashed_sizes: List[int],
    input_numel: int,
) -> List[float]:
    """Estimate privacy leakage for each candidate split point.

    Following HSFL, privacy leakage is proportional to the amount of
    information in the intermediate tensor relative to the original input.
    A larger intermediate is closer to the raw data and reveals more.

    We normalise by the input numel so that leakage ∈ [0, 1].
    """
    if input_numel <= 0:
        input_numel = max(smashed_sizes) if smashed_sizes else 1
    leakages = []
    for sz in smashed_sizes:
        leakages.append(min(float(sz) / float(input_numel), 1.0))
    return leakages


class SplitPointSelector:
    """Adaptive split-point selector using a **LinUCB contextual bandit**
    (as in HSFL, ICWS 2023).

    The selector maintains a set of *candidate* split points, each
    characterised by a context vector ``[cumulative_flops, smashed_data_size]``.
    At each round it picks the action (split-point) that minimises an
    estimated cost, balancing *latency* (from the UCB estimate) and *privacy
    leakage* (a static score per split point).

    After each round the caller feeds back the observed latency, which
    updates the LinUCB model.

    Parameters
    ----------
    profiles : list[LayerProfile]
        One profile per candidate split point.
    latency_weight : float
        Trade-off coefficient.  ``cost = latency_weight * UCB + (1-latency_weight) * privacy``.
        Default 1.0 → pure latency minimisation (no privacy).
    alpha : float
        UCB exploration bonus coefficient (higher → more exploration).
    """

    def __init__(
        self,
        profiles: List[LayerProfile],
        *,
        latency_weight: float = 1.0,
        alpha: float = 0.25,
    ):
        if not profiles:
            raise ValueError("At least one candidate split point is required.")

        self.profiles = list(profiles)
        self.num_actions = len(self.profiles)
        self.action_indices = [p.index for p in self.profiles]

        # LinUCB context dimension: [cumulative_flops, smashed_data_size]
        self._ctx_dim = 2
        self._x_theta = np.zeros((self._ctx_dim, self.num_actions), dtype=np.float64)
        for i, p in enumerate(self.profiles):
            self._x_theta[0][i] = p.cumulative_flops
            self._x_theta[1][i] = p.smashed_data_size

        # Normalise so features are on similar scales
        for d in range(self._ctx_dim):
            row_max = np.max(np.abs(self._x_theta[d]))
            if row_max > 0:
                self._x_theta[d] /= row_max

        self._privacy = np.array(
            [p.privacy_leakage for p in self.profiles], dtype=np.float64,
        )

        self.alpha = alpha
        self.latency_weight = latency_weight

        # LinUCB state
        self._A = np.eye(self._ctx_dim, dtype=np.float64)
        self._b = np.zeros((self._ctx_dim, 1), dtype=np.float64)

        self._history: List[dict] = []

    # ---- public API -------------------------------------------------

    def select(self) -> int:
        """Choose the best split-point index using LinUCB.

        Returns the *layer index* (in the traced graph) of the selected
        split point.
        """
        A_inv = np.linalg.inv(self._A)
        theta = A_inv @ self._b          # (ctx_dim, 1)

        costs: List[float] = []
        for i in range(self.num_actions):
            x_i = self._x_theta[:, [i]]                   # (ctx_dim, 1)
            exploit = float(x_i.T @ theta)                 # estimated latency
            explore = float(self.alpha * np.sqrt(x_i.T @ A_inv @ x_i))
            # Lower is better: exploitation minus exploration bonus
            cost = self.latency_weight * (exploit - explore) + \
                   (1.0 - self.latency_weight) * self._privacy[i]
            costs.append(cost)

        best_idx = int(np.argmin(costs))
        return self.action_indices[best_idx]

    def update(self, selected_layer_index: int, observed_latency: float) -> None:
        """Feed back the observed training latency after one round.

        Parameters
        ----------
        selected_layer_index : int
            The layer index that was used (as returned by ``select``).
        observed_latency : float
            Measured wall-clock training time for that round.
        """
        try:
            action_pos = self.action_indices.index(selected_layer_index)
        except ValueError:
            logger.warning(
                "[SplitSelector] Layer {} not in candidate list — skipping update.",
                selected_layer_index,
            )
            return

        x_i = self._x_theta[:, [action_pos]]
        self._A += x_i @ x_i.T
        self._b += x_i * observed_latency

        self._history.append({
            "layer_index": selected_layer_index,
            "observed_latency": observed_latency,
        })

    def select_by_flops_ratio(self, target_ratio: float) -> int:
        """Select the split point whose cumulative FLOPs ratio is closest
        to ``target_ratio`` ∈ [0, 1].

        This mirrors HSFL's ``action_to_layer()`` helper (used by the RL
        agent): it converts a continuous offloading value into a discrete
        layer index based on cumulative FLOPs.

        Parameters
        ----------
        target_ratio : float
            Desired fraction of total FLOPs to execute on the edge.
            0.0 → split very early (almost everything on cloud),
            1.0 → split at the end (everything on edge = no offloading).

        Returns
        -------
        int — layer index of the closest candidate.
        """
        total_flops = max(p.cumulative_flops for p in self.profiles) if self.profiles else 1.0
        ratios = np.array(
            [p.cumulative_flops / total_flops for p in self.profiles],
            dtype=np.float64,
        )
        best = int(np.argmin(np.abs(ratios - target_ratio)))
        return self.action_indices[best]

    def select_by_privacy(self, max_leakage: float = 0.1) -> int:
        """Select the deepest split point (most FLOPs on edge) whose
        privacy leakage does not exceed ``max_leakage``.

        Parameters
        ----------
        max_leakage : float
            Maximum acceptable privacy-leakage score (0–1).

        Returns
        -------
        int — layer index.
        """
        valid = [(i, p) for i, p in enumerate(self.profiles) if p.privacy_leakage <= max_leakage]
        if not valid:
            logger.warning(
                "[SplitSelector] No candidate has leakage ≤ {}. Using least-leakage point.",
                max_leakage,
            )
            best = int(np.argmin(self._privacy))
            return self.action_indices[best]
        # Among valid, pick the one with the most edge-side FLOPs
        best_action_pos = max(valid, key=lambda t: t[1].cumulative_flops)[0]
        return self.action_indices[best_action_pos]

    def get_profiles(self) -> List[LayerProfile]:
        """Return the list of candidate profiles (read-only copy)."""
        return list(self.profiles)

    def __repr__(self) -> str:
        return (
            f"SplitPointSelector(candidates={self.num_actions}, "
            f"alpha={self.alpha}, latency_w={self.latency_weight})"
        )


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
                vis_opt="none",
                layers_to_save="all",
                save_function_args=save_function_args,
            )

        self._model_history = mh
        self._layer_list = list(mh.layer_list)  # shallow copy of the list
        self._label2idx = _prepare_replay_graph(self._layer_list)
        self._traced = True
        self._split_index = None

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
            infos.append(
                LayerInfo(
                    index=i,
                    label=getattr(layer, "layer_label", f"layer_{i}"),
                    layer_type=getattr(layer, "layer_type", "unknown"),
                    func_name=getattr(layer, "func_applied_name", ""),
                    output_shape=getattr(layer, "tensor_shape", None),
                    has_params=bool(getattr(layer, "computed_with_params", False)),
                    num_params=getattr(layer, "num_params_total", 0) or 0,
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
    # 3b. HSFL-style layer profiling & adaptive split-point selection
    # ------------------------------------------------------------------

    def profile_layers(self) -> List[LayerProfile]:
        """Profile **every** traced layer with FLOPs and smashed-data-size.

        Returns a list of ``LayerProfile`` — one per layer — containing
        cumulative FLOPs, intermediate tensor numel, and a privacy-leakage
        heuristic.  This is the universalised version of HSFL's
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
        HSFL's ``actionList`` which only contains meaningful split-point layer
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
            if only_parametric and not getattr(layer, "computed_with_params", False):
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

    def create_split_selector(
        self,
        *,
        only_parametric: bool = False,
        latency_weight: float = 1.0,
        alpha: float = 0.25,
    ) -> SplitPointSelector:
        """Create a ``SplitPointSelector`` (LinUCB bandit) from the traced
        model's layer profiles.

        This is the recommended way to perform HSFL-style adaptive split-point
        selection.  Usage::

            selector = splitter.create_split_selector()
            layer_idx = selector.select()       # initial choice
            splitter.split(layer_index=layer_idx)
            ...
            # after training round, feed back the measured latency
            selector.update(layer_idx, observed_latency)
            # next round
            layer_idx = selector.select()

        Parameters
        ----------
        only_parametric : bool
            Only include parametric layers as candidates.
        latency_weight : float
            Trade-off ``latency`` vs ``privacy`` (1.0 = pure latency).
        alpha : float
            LinUCB exploration coefficient.

        Returns
        -------
        SplitPointSelector
        """
        candidates = self.get_candidate_split_points(only_parametric=only_parametric)
        return SplitPointSelector(
            candidates,
            latency_weight=latency_weight,
            alpha=alpha,
        )

    def select_split_point(
        self,
        *,
        strategy: str = "flops_ratio",
        target_ratio: float = 0.5,
        max_privacy_leakage: float = 0.1,
        only_parametric: bool = False,
    ) -> int:
        """One-shot convenience: select **and set** the split point.

        Parameters
        ----------
        strategy : str
            ``'flops_ratio'`` — split so that ~``target_ratio`` of total
              FLOPs run on the edge (HSFL ``action_to_layer``).
            ``'privacy'`` — deepest split where privacy leakage ≤
              ``max_privacy_leakage``.
            ``'midpoint'`` — split at the candidate closest to 50 % FLOPs.
            ``'min_smashed'`` — split where the intermediate tensor is
              smallest (minimises communication).
        target_ratio : float
            For ``'flops_ratio'`` strategy.
        max_privacy_leakage : float
            For ``'privacy'`` strategy.
        only_parametric : bool
            Only consider parametric layers as candidates.

        Returns
        -------
        int — the chosen layer index (also stored via ``self.split()``).
        """
        candidates = self.get_candidate_split_points(only_parametric=only_parametric)
        selector = SplitPointSelector(candidates)

        if strategy == "flops_ratio":
            idx = selector.select_by_flops_ratio(target_ratio)
        elif strategy == "privacy":
            idx = selector.select_by_privacy(max_privacy_leakage)
        elif strategy == "midpoint":
            idx = selector.select_by_flops_ratio(0.5)
        elif strategy == "min_smashed":
            best = min(candidates, key=lambda p: p.smashed_data_size)
            idx = best.index
        else:
            raise ValueError(
                f"Unknown strategy {strategy!r}. "
                f"Use 'flops_ratio', 'privacy', 'midpoint', or 'min_smashed'."
            )

        self.split(layer_index=idx)
        logger.info(
            "[UniversalSplit] Auto-selected split point via '{}': layer {} (index {}).",
            strategy,
            self._layer_list[idx].layer_label,
            idx,
        )
        return idx

    # ------------------------------------------------------------------
    # 4. Edge-side forward (head partition)
    # ------------------------------------------------------------------

    def edge_forward(
        self,
        x: torch.Tensor,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
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
        Tensor — intermediate activations (detached, ready for transmission).
        """
        self._ensure_split()
        dev = torch.device(device) if device else self.device

        if dev and x.device != dev:
            x = x.to(dev)

        _set_training_flag(self._layer_list, training=False)
        with torch.no_grad():
            _, intermediate = _partition_forward_head(
                self._layer_list,
                self._label2idx,
                x,
                self._split_index,
            )
        return intermediate

    # ------------------------------------------------------------------
    # 5. Cloud-side forward (tail partition) — inference
    # ------------------------------------------------------------------

    def cloud_forward(
        self,
        intermediate: torch.Tensor,
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

        if dev and intermediate.device != dev:
            intermediate = intermediate.to(dev)

        _set_training_flag(self._layer_list, training=False)
        with torch.no_grad():
            output = _partition_forward_tail(
                self._layer_list,
                self._label2idx,
                intermediate,
                self._split_index,
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

        _set_training_flag(self._layer_list, training=True)

        output = _partition_forward_tail_train(
            self._layer_list,
            self._label2idx,
            intermediate,
            self._split_index,
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
        _freeze_head_params(self._layer_list, self._split_index)

    def unfreeze_tail(self) -> None:
        """Unfreeze all parameters in the tail (cloud) partition."""
        self._ensure_split()
        _unfreeze_tail_params(self._layer_list, self._split_index)

    def get_tail_trainable_params(self) -> list[torch.Tensor]:
        """Return a list of trainable parameters in the cloud tail."""
        self._ensure_split()
        return _get_trainable_params(self._layer_list, self._split_index)

    def get_all_params(self) -> list[torch.Tensor]:
        """Return all parameters of the full model."""
        self._ensure_traced()
        return _get_all_params(self._layer_list)

    def load_weights(self, weights: list[torch.Tensor]) -> None:
        self._ensure_traced()
        _load_weights_into_layers(self._layer_list, weights)

    # ------------------------------------------------------------------
    # 8. Serialisation for network transmission
    # ------------------------------------------------------------------

    @staticmethod
    def serialise_intermediate(
        tensor: torch.Tensor,
        *,
        compress: bool = False,
    ) -> bytes:
        """Serialise an intermediate tensor for network transfer.

        Parameters
        ----------
        tensor : Tensor
        compress : bool
            If ``True``, apply zlib compression.

        Returns
        -------
        bytes — serialised payload.
        """
        buf = io.BytesIO()
        torch.save(tensor.detach().cpu(), buf)
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
    ) -> torch.Tensor:
        """Deserialise an intermediate tensor received over the network.

        Parameters
        ----------
        data   : bytes
        device : target device
        compressed : whether zlib decompression is needed

        Returns
        -------
        Tensor on *device*.
        """
        if compressed:
            import zlib
            data = zlib.decompress(data)
        buf = io.BytesIO(data)
        tensor = torch.load(buf, map_location=device)
        return tensor

    # ------------------------------------------------------------------
    # 9. Full inference (unsplit) via replay
    # ------------------------------------------------------------------

    def full_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the full model layer-by-layer (no split).  Useful for
        validating that the traced graph is faithful.
        """
        self._ensure_traced()
        _set_training_flag(self._layer_list, training=False)
        with torch.no_grad():
            return _layer_forward_full(self._layer_list, self._label2idx, x)

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
            params = getattr(layer, "parent_params", None)
            if params:
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
            if layer.parent_params and param_idx < len(layer.parent_params):
                layer.parent_params[param_idx] = val

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
            except Exception:
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
        except Exception:
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
            except Exception as exc:
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
    return torch.load(path, map_location="cpu")
