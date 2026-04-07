"""
Dynamic Activation Sparsity (DAS) — Cloud-Side Split-Model Continual Learning
==============================================================================

Reference
---------
SURGEON (CVPR 2025) — *"Memory-Adaptive Fully Test-Time Adaptation
via Dynamic Activation Sparsity"*
<https://github.com/kadmkbl/SURGEON>

Core mechanism
--------------
During cloud-side continual learning of the split model's **tail partition**,
activations cached for gradient computation in the backward pass are
*dynamically pruned* on a per-layer basis.  Per-layer pruning ratios are
computed adaptively using:

*  **Gradient Importance (GI)** — ``||grad|| / sqrt(numel)``
*  **Memory Importance (MI)** — ``log(M_total / M_layer)``
*  **Total Gradient Importance (TGI)** — ``GI × MI``

Layers with higher TGI are deemed more important for training and keep
more activations (lower pruning ratio).  This reduces GPU memory
consumption while maintaining training performance.

Two-phase training step
-----------------------
1. **Probe phase** — disable sparsity, run a small-batch forward-backward
   to collect per-layer gradients and compute TGI.
2. **Train phase** — enable sparsity with the computed per-layer pruning
   ratios, perform the main forward-backward-update step.

Integration with the Plank-road pipeline
-----------------------------------------
``DASTrainer`` wraps any ``nn.Module`` (typically the tail partition,
e.g. detector-tail modules such as ``head`` or ``roi_heads``) and transparently manages
module replacement and pruning-ratio computation.  It is called from
``model_split.split_retrain()`` and ``universal_split_retrain()`` when
the ``das_enabled`` flag is set in the configuration.
"""

from __future__ import annotations

import collections
import math
from collections import defaultdict
from itertools import repeat
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import (
    check_backward_validity,
    get_device_states,
    set_device_states,
)
from loguru import logger


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Activation Clipper — sparse activation storage for backpropagation
# ═══════════════════════════════════════════════════════════════════════════

class ActivationClipper:
    """Prunes activation tensors by keeping only the top-k elements by
    absolute value, reducing backward-pass memory consumption.

    The *clip* method flattens the tensor, selects the ``ceil(numel × (1 -
    clip_ratio))`` largest-magnitude elements and their flat indices, then
    discards the rest.  *reshape* scatters the kept values back into a
    zero-filled tensor of the original shape for use in gradient computation.

    Parameters
    ----------
    clip_ratio : float
        Fraction of elements to prune.  ``0`` keeps all, ``1`` prunes all.
    """

    def __init__(self, clip_ratio: float):
        self.clip_ratio = max(0.0, min(clip_ratio, 1.0))

    # ---- clip & reshape (min-abs mode) ----------------------------------

    def clip(self, x: torch.Tensor, ctx) -> torch.Tensor:
        """Flatten *x*, keep top-k by ``|val|``, save indices in *ctx*."""
        ctx.x_shape = x.shape
        numel = x.numel()
        ctx.numel = numel
        x_flat = x.reshape(-1)

        if self.clip_ratio <= 0 or numel == 0:
            ctx.idxs = None
            return x_flat.clone()

        keep_n = max(1, int(numel * (1.0 - self.clip_ratio)))
        idxs = x_flat.abs().topk(keep_n, sorted=False)[1]
        clipped = x_flat[idxs]
        ctx.idxs = idxs.to(torch.int32)
        return clipped

    def reshape(self, x: torch.Tensor, ctx) -> torch.Tensor:
        """Scatter *x* back into a zero-filled tensor with original shape."""
        if ctx.idxs is None:
            return x
        idxs = ctx.idxs.to(torch.int64)
        del ctx.idxs
        full = torch.zeros(ctx.numel, device=x.device, dtype=x.dtype)
        full.scatter_(0, idxs, x)
        return full


# helper
def _pair(x):
    """Convert *x* to a 2-tuple if not already iterable."""
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return (x, x)


def _spectral_entropy_1d(x: torch.Tensor, *, max_samples: int = 2048, eps: float = 1e-12) -> float | None:
    """Compute normalised spectral entropy of a 1D signal.

    Returns a value in [0, 1] (higher => more complex spectrum).
    """
    if x.numel() == 0:
        return None
    flat = x.reshape(-1)
    if flat.numel() > max_samples:
        idx = torch.randperm(flat.numel(), device=flat.device)[:max_samples]
        flat = flat[idx]
    flat = flat.float()
    spectrum = torch.fft.rfft(flat)
    power = spectrum.abs().pow(2)
    total = power.sum()
    if total.item() <= 0:
        return 0.0
    probs = power / total
    entropy = -(probs * (probs + eps).log()).sum()
    norm = math.log(max(int(probs.numel()), 1))
    if norm <= 0:
        return 0.0
    return float((entropy / norm).clamp(0.0, 1.0).item())


# ═══════════════════════════════════════════════════════════════════════════
# 2.  AutoFreezeConv2d — Conv2d with Dynamic Activation Sparsity
# ═══════════════════════════════════════════════════════════════════════════

class AutoFreezeConv2d(nn.Conv2d):
    """Drop-in replacement for ``nn.Conv2d`` that supports per-layer
    Dynamic Activation Sparsity.

    When ``sparsity_signal`` is *True* and ``clip_ratio > 0``, the input
    activations stored for backward are pruned (smallest magnitudes
    zeroed), reducing peak GPU memory.

    Extra attributes compared to ``nn.Conv2d``:

    ============== ======================================================
    ``name``       human-readable layer name
    ``clip_ratio`` fraction of activations to prune (0 → no pruning)
    ``sparsity_signal`` DAS enabled flag
    ``activation_size`` last forward's input numel (for memory profiling)
    ``back_cache_size`` numel actually stored for backward
    ``bn_only``    if *True*, weight gradient is blocked in backward
    ============== ======================================================
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        *,
        name: str = "conv",
        num: int = 0,
        bn_only: bool = False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=True if (bias is True or (bias is not None and bias is not False)) else False,
            padding_mode=padding_mode,
        )
        self.name = name
        self.num = num
        self.clip_ratio: float = 0.0
        self.sparsity_signal: bool = False
        self.back_cache_size: int = 0
        self.activation_size: int = 0
        self.bn_only: bool = bn_only
        self.track_spectral_entropy: bool = False
        self.last_spectral_entropy: float | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return _AutoFreezeConv2dFn.apply(self, x, self.weight, self.bias)

    def conv_forward(self, x: torch.Tensor, weight, bias) -> torch.Tensor:
        """Standard convolution forward (no custom autograd)."""
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                weight,
                bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)


class _AutoFreezeConv2dFn(torch.autograd.Function):
    """Autograd function for Conv2d with activation sparsity."""

    @staticmethod
    def forward(ctx, module: AutoFreezeConv2d, x, weight, bias):
        check_backward_validity([x, weight] + ([bias] if bias is not None else []))
        ctx.module = module

        # --- RNG state bookkeeping (for reproducibility) -----------------
        ctx.fwd_cpu_state = torch.get_rng_state()
        ctx.had_cuda_in_fwd = torch.cuda._initialized
        if ctx.had_cuda_in_fwd:
            ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(x)

        with torch.no_grad():
            y = module.conv_forward(x, weight, bias)

            # Track activation size for TGI memory importance
            module.activation_size = int(x.numel())
            if module.track_spectral_entropy:
                module.last_spectral_entropy = _spectral_entropy_1d(x.detach())

            # Dynamic Activation Sparsity
            if module.sparsity_signal and module.clip_ratio > 0:
                clipper = ActivationClipper(module.clip_ratio)
                module.back_cache_size = int(x.numel() * (1.0 - module.clip_ratio))
            else:
                clipper = ActivationClipper(0)
                module.back_cache_size = int(x.numel())

            ctx.clipper = clipper
            clipped_x = clipper.clip(x, ctx)
            ctx.save_for_backward(clipped_x)

        return y

    @staticmethod
    def backward(ctx, grad_out):
        module = ctx.module
        clipper = ctx.clipper
        (clipped_x,) = ctx.saved_tensors

        # Restore RNG state
        rng_devices = []
        if ctx.had_cuda_in_fwd:
            rng_devices = ctx.fwd_gpu_devices
        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(ctx.fwd_cpu_state)
            if ctx.had_cuda_in_fwd:
                set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)

            # Reconstruct input from clipped values
            x_restored = clipper.reshape(clipped_x, ctx).view(ctx.x_shape)

            # Compute grad_x and grad_w via built-in convolution_backward
            grad_x, grad_w = torch.ops.aten.convolution_backward(
                grad_out,
                x_restored,
                module.weight,
                None,
                module.stride,
                module.padding,
                module.dilation,
                False,
                [0],
                module.groups,
                (True, True, False),
            )[:2]

            del clipper

        if module.bn_only:
            # BN-only mode: block weight gradient
            return None, grad_x, None, None
        else:
            return None, grad_x, grad_w, None


# ═══════════════════════════════════════════════════════════════════════════
# 3.  DASBatchNorm2d — BatchNorm2d with DAS metadata tracking
# ═══════════════════════════════════════════════════════════════════════════

class DASBatchNorm2d(nn.BatchNorm2d):
    """Drop-in replacement for ``nn.BatchNorm2d`` with Dynamic Activation
    Sparsity metadata tracking.

    Unlike Conv and FC layers, BN activation tensors are comparatively
    small (they scale linearly with channels, not quadratically with
    spatial dimensions).  This implementation tracks ``activation_size``
    for TGI memory importance computation but delegates the forward /
    backward to standard PyTorch BN, keeping the implementation simple.

    When ``sparsity_signal`` is ``True`` **and** ``clip_ratio > 0``, a
    custom autograd path with activation pruning is used (faithful to
    SURGEON's ``AutoFreezeNorm2d``), computing ``grad_x`` from the
    **full** input and ``grad_w`` from the **clipped** input.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        *,
        name: str = "bn",
        num: int = 0,
        bn_only: bool = False,
    ):
        super().__init__(num_features, eps=eps, momentum=momentum, affine=affine,
                         track_running_stats=track_running_stats)
        self.name = name
        self.num = num
        self.clip_ratio: float = 0.0
        self.sparsity_signal: bool = False
        self.back_cache_size: int = 0
        self.activation_size: int = 0
        self.bn_only: bool = bn_only
        self.track_spectral_entropy: bool = False
        self.last_spectral_entropy: float | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.activation_size = int(x.numel())
        if self.track_spectral_entropy:
            self.last_spectral_entropy = _spectral_entropy_1d(x.detach())
        if self.sparsity_signal and self.clip_ratio > 0:
            return _DASBatchNorm2dFn.apply(self, x, self.weight, self.bias)
        # Fast path: standard BN (no sparsity)
        return super().forward(x)

    def _bn_forward_for_backward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        batch_mean: torch.Tensor,
        batch_var: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterised BN for backward gradient computation (SURGEON).

        Converts training-mode BN into an equivalent form using running
        statistics, expressed as a reparameterised training-mode BN so
        that autograd can track the computation.
        """
        with torch.no_grad():
            inv_r_std = torch.sqrt(running_var + self.eps)
            weight_hat = torch.sqrt(batch_var + self.eps) / inv_r_std
            bias_hat = (batch_mean - running_mean) / inv_r_std

        weight_hat = weight * weight_hat
        bias_hat = weight * bias_hat + bias

        return F.batch_norm(x, None, None, weight_hat, bias_hat,
                            training=True, momentum=0.0, eps=self.eps)


class _DASBatchNorm2dFn(torch.autograd.Function):
    """Autograd for BN with activation sparsity (SURGEON-style)."""

    @staticmethod
    def forward(ctx, module: DASBatchNorm2d, x, weight, bias):
        check_backward_validity([x] + ([weight] if weight is not None else [])
                                + ([bias] if bias is not None else []))
        ctx.module = module
        ctx.x_full = x.clone()  # full input for accurate grad_x

        # RNG state
        ctx.fwd_cpu_state = torch.get_rng_state()
        ctx.had_cuda_in_fwd = torch.cuda._initialized
        if ctx.had_cuda_in_fwd:
            ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(x)

        with torch.no_grad():
            # Capture batch statistics
            batch_var, batch_mean = torch.var_mean(x, dim=(0, 2, 3), unbiased=False)
            running_mean = module.running_mean.clone()
            running_var = module.running_var.clone()

            # Standard BN forward
            y = F.batch_norm(x, module.running_mean, module.running_var,
                             weight, bias, module.training,
                             module.momentum, module.eps)

            module.activation_size = int(x.numel())

            # DAS clipping for weight gradient
            clipper = ActivationClipper(module.clip_ratio)
            module.back_cache_size = int(x.numel() * (1.0 - module.clip_ratio))
            ctx.clipper = clipper
            clipped_x = clipper.clip(x, ctx)
            clipped_x.requires_grad = x.requires_grad

            ctx.save_for_backward(clipped_x, batch_mean, batch_var,
                                  running_mean, running_var)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        module = ctx.module
        clipper = ctx.clipper
        c_x, batch_mean, batch_var, running_mean, running_var = ctx.saved_tensors

        # Restore RNG
        rng_devices = []
        if ctx.had_cuda_in_fwd:
            rng_devices = ctx.fwd_gpu_devices
        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            if ctx.had_cuda_in_fwd:
                set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)

            # Full input → accurate grad_x
            detached_x = ctx.x_full.detach()
            detached_x.requires_grad = c_x.requires_grad
            if module.affine:
                weight = module.weight.detach().requires_grad_(module.weight.requires_grad)
                bias = module.bias.detach().requires_grad_(module.bias.requires_grad)
                weight_c = module.weight.detach().requires_grad_(module.weight.requires_grad)
                bias_c = module.bias.detach().requires_grad_(module.bias.requires_grad)
            else:
                weight = bias = weight_c = bias_c = None

            # Clipped input → approximate grad_w
            clipped_x = clipper.reshape(c_x, ctx).view(ctx.x_shape)
            clipped_x.requires_grad = c_x.requires_grad

            with torch.enable_grad():
                y = module._bn_forward_for_backward(
                    detached_x, weight, bias,
                    running_mean, running_var, batch_mean, batch_var,
                )
                c_y = module._bn_forward_for_backward(
                    clipped_x, weight_c, bias_c,
                    running_mean, running_var, batch_mean, batch_var,
                )

            with torch.no_grad():
                if torch.is_tensor(y) and y.requires_grad:
                    torch.autograd.backward([y], [grad_out])
                if torch.is_tensor(c_y) and c_y.requires_grad:
                    torch.autograd.backward([c_y], [grad_out])

            grad_x = detached_x.grad
            grad_w = weight_c.grad if module.affine and weight_c is not None else None

            del clipper

        return None, grad_x, grad_w, None


# ═══════════════════════════════════════════════════════════════════════════
# 4.  AutoFreezeFC — Linear with Dynamic Activation Sparsity
# ═══════════════════════════════════════════════════════════════════════════

class AutoFreezeFC(nn.Linear):
    """Drop-in replacement for ``nn.Linear`` with Dynamic Activation
    Sparsity, mirroring ``AutoFreezeConv2d``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        name: str = "fc",
        num: int = 0,
        bn_only: bool = False,
    ):
        super().__init__(
            in_features,
            out_features,
            True if (bias is True or (bias is not None and bias is not False)) else False,
        )
        self.name = name
        self.num = num
        self.clip_ratio: float = 0.0
        self.sparsity_signal: bool = False
        self.back_cache_size: int = 0
        self.activation_size: int = 0
        self.bn_only: bool = bn_only
        self.track_spectral_entropy: bool = False
        self.last_spectral_entropy: float | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return _AutoFreezeFCFn.apply(self, x, self.weight, self.bias)

    def fc_forward(self, x, weight, bias):
        return F.linear(x, weight, bias)


class _AutoFreezeFCFn(torch.autograd.Function):
    """Autograd function for Linear with activation sparsity."""

    @staticmethod
    def forward(ctx, module: AutoFreezeFC, x, weight, bias):
        check_backward_validity([x, weight] + ([bias] if bias is not None else []))
        ctx.module = module

        ctx.fwd_cpu_state = torch.get_rng_state()
        ctx.had_cuda_in_fwd = torch.cuda._initialized
        if ctx.had_cuda_in_fwd:
            ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(x)

        with torch.no_grad():
            y = module.fc_forward(x, weight, bias)

            module.activation_size = int(x.numel())
            if module.track_spectral_entropy:
                module.last_spectral_entropy = _spectral_entropy_1d(x.detach())

            if module.sparsity_signal and module.clip_ratio > 0:
                clipper = ActivationClipper(module.clip_ratio)
                module.back_cache_size = int(x.numel() * (1.0 - module.clip_ratio))
            else:
                clipper = ActivationClipper(0)
                module.back_cache_size = int(x.numel())

            ctx.clipper = clipper
            clipped_x = clipper.clip(x, ctx)
            ctx.save_for_backward(clipped_x)

        return y

    @staticmethod
    def backward(ctx, grad_out):
        module = ctx.module
        clipper = ctx.clipper
        (clipped_x,) = ctx.saved_tensors

        rng_devices = []
        if ctx.had_cuda_in_fwd:
            rng_devices = ctx.fwd_gpu_devices
        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(ctx.fwd_cpu_state)
            if ctx.had_cuda_in_fwd:
                set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)

            x_restored = clipper.reshape(clipped_x, ctx)

            # Linear: weight.shape = (out, in)
            ic, oc = module.weight.shape  # ic=out_features, oc=in_features
            grad_w = grad_out.reshape(-1, ic).T.mm(x_restored.reshape(-1, oc))
            grad_x = torch.matmul(grad_out, module.weight)

            del clipper

        if module.bn_only:
            return None, grad_x, None, None
        else:
            return None, grad_x, grad_w, None


# ═══════════════════════════════════════════════════════════════════════════
# 5.  TGI — Total Gradient Importance (gradient × memory)
# ═══════════════════════════════════════════════════════════════════════════

def compute_tgi(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    layer_names: list[str],
    layer_memories: list[float],
    memory_sum: float,
) -> dict[str, float]:
    """Per-layer Total Gradient Importance (SURGEON / ICWS-23 inspired).

    TGI_l  =  (||grad_l|| / sqrt(|grad_l|))  ×  log(M_total / M_l)

    Layers with **higher** TGI retain **more** activations (lower pruning).

    Parameters
    ----------
    params : list[Tensor]
        Model parameters (with gradients).
    grads : list[Tensor]
        Corresponding gradient tensors.
    layer_names : list[str]
        Parameter names.
    layer_memories : list[float]
        Per-layer activation sizes (``numel``).
    memory_sum : float
        Total activation memory across all layers.

    Returns
    -------
    dict mapping layer name → TGI score.
    """
    metrics: dict[str, float] = {}
    for name, param, grad, mem in zip(layer_names, params, grads, layer_memories):
        grad_norm = torch.norm(grad).item()
        gi = grad_norm / max(grad.numel() ** 0.5, 1.0)  # Gradient Importance
        mi = math.log(max(memory_sum, 1.0) / max(mem, 1.0))  # Memory Importance
        metrics[name] = gi * mi
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# 6.  DASTrainer — high-level API for DAS-enabled training
# ═══════════════════════════════════════════════════════════════════════════

class DASTrainer:
    """Manages Dynamic Activation Sparsity for any ``nn.Module``.

    Typical usage::

        trainer = DASTrainer(model, device="cuda")
        # optionally probe to set per-layer pruning ratios
        trainer.probe_and_set_ratios(sample_input, loss_fn)
        # enable DAS
        trainer.activate_sparsity()
        # … normal training loop, DAS is transparent …
        output = model(x)
        loss.backward()
        optimizer.step()
        # optionally update ratios periodically
        trainer.probe_and_set_ratios(new_sample, loss_fn)

    Parameters
    ----------
    model : nn.Module
        The model (or tail partition) whose modules will be replaced.
    bn_only : bool
        If *True*, weight gradients of Conv/FC layers are blocked
        (only BN parameters are updated).
    tau : float
        Temperature controlling pruning-ratio uniformity.
        Higher → more uniform ratios.  Not used in current implementation
        (reserved for future softmax-temperature scaling).
    probe_samples : int
        Number of samples for gradient-importance probing.
    device : str | torch.device
        Computation device.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        bn_only: bool = False,
        tau: float = 0.5,
        probe_samples: int = 10,
        use_spectral_entropy: bool = False,
        device: str | torch.device = "cpu",
    ):
        self.model = model
        self.bn_only = bn_only
        self.tau = tau
        self.probe_samples = probe_samples
        self.device = torch.device(device)
        self._pruning_ratios: dict[str, float] = {}
        self.use_spectral_entropy = use_spectral_entropy

        # Replace modules with AutoFreeze versions
        self._n_conv = self._replace_conv(model, model.__class__.__name__)
        self._n_bn = self._replace_bn(model, model.__class__.__name__)
        self._n_fc = self._replace_fc(model, model.__class__.__name__)
        logger.info(
            "[DAS] Replaced {} Conv + {} BN + {} FC layers with AutoFreeze versions.",
            self._n_conv, self._n_bn, self._n_fc,
        )
        if self.use_spectral_entropy:
            self.enable_spectral_entropy_tracking(True)

    # ------------------------------------------------------------------
    # Module replacement (Conv → AutoFreezeConv2d, BN → DASBatchNorm2d,
    # Linear → AutoFreezeFC)
    # ------------------------------------------------------------------

    def _replace_conv(self, model: nn.Module, name: str, count: int = 0) -> int:
        copy_keys = ["stride", "padding", "dilation", "groups", "bias", "padding_mode"]
        for mod_name, target_mod in model.named_children():
            if isinstance(target_mod, nn.Conv2d) and not isinstance(target_mod, AutoFreezeConv2d):
                count += 1
                new_mod = AutoFreezeConv2d(
                    target_mod.in_channels,
                    target_mod.out_channels,
                    target_mod.kernel_size,
                    **{k: getattr(target_mod, k) for k in copy_keys},
                    name=f"{name}.{mod_name}",
                    num=count,
                    bn_only=self.bn_only,
                )
                new_mod.load_state_dict(target_mod.state_dict())
                setattr(model, mod_name, new_mod)
            else:
                count = self._replace_conv(target_mod, f"{name}.{mod_name}", count)
        return count

    def _replace_bn(self, model: nn.Module, name: str, count: int = 0) -> int:
        copy_keys = ["eps", "momentum", "affine", "track_running_stats"]
        for mod_name, target_mod in model.named_children():
            if isinstance(target_mod, nn.BatchNorm2d) and not isinstance(target_mod, DASBatchNorm2d):
                count += 1
                new_mod = DASBatchNorm2d(
                    target_mod.num_features,
                    **{k: getattr(target_mod, k) for k in copy_keys},
                    name=f"{name}.{mod_name}",
                    num=count,
                    bn_only=self.bn_only,
                )
                new_mod.load_state_dict(target_mod.state_dict())
                setattr(model, mod_name, new_mod)
            else:
                count = self._replace_bn(target_mod, f"{name}.{mod_name}", count)
        return count

    def _replace_fc(self, model: nn.Module, name: str, count: int = 0) -> int:
        for mod_name, target_mod in model.named_children():
            if isinstance(target_mod, nn.Linear) and not isinstance(target_mod, AutoFreezeFC):
                count += 1
                new_mod = AutoFreezeFC(
                    target_mod.in_features,
                    target_mod.out_features,
                    target_mod.bias is not None,
                    name=f"{name}.{mod_name}",
                    num=count,
                    bn_only=self.bn_only,
                )
                new_mod.load_state_dict(target_mod.state_dict())
                setattr(model, mod_name, new_mod)
            else:
                count = self._replace_fc(target_mod, f"{name}.{mod_name}", count)
        return count

    # ------------------------------------------------------------------
    # AutoFreeze module enumeration
    # ------------------------------------------------------------------

    def _das_modules(self) -> dict[str, nn.Module]:
        """Return all AutoFreeze / DAS modules keyed by name."""
        modules: dict[str, nn.Module] = {}
        for name, mod in self.model.named_modules():
            if isinstance(mod, (AutoFreezeConv2d, DASBatchNorm2d, AutoFreezeFC)):
                modules[name] = mod
        return modules

    def enable_spectral_entropy_tracking(self, enabled: bool = True) -> None:
        """Enable/disable spectral entropy tracking on DAS modules."""
        for _, mod in self._das_modules().items():
            mod.track_spectral_entropy = bool(enabled)

    def _spectral_entropy_scores(self) -> dict[str, float]:
        """Return per-layer spectral-entropy scores keyed by param name."""
        scores: dict[str, float] = {}
        mem_dict, mem_sum = self._get_layer_memories()
        for name, mod in self.model.named_modules():
            if not isinstance(mod, (AutoFreezeConv2d, DASBatchNorm2d, AutoFreezeFC)):
                continue
            entropy = getattr(mod, "last_spectral_entropy", None)
            if entropy is None:
                continue
            key = name + ".weight"
            mem = mem_dict.get(key, 1.0)
            mi = math.log(max(mem_sum, 1.0) / max(mem, 1.0))
            scores[key] = float(entropy) * float(mi)
        return scores

    def refresh_pruning_ratios_from_spectral_entropy(self) -> dict[str, float]:
        """Compute pruning ratios from spectral entropy (no gradients)."""
        scores = self._spectral_entropy_scores()
        if not scores:
            self._pruning_ratios = {}
            return {}
        max_score = max(scores.values()) or 1.0
        pruning_ratios = {k: max(0.0, 1.0 - v / max_score) for k, v in scores.items()}
        self._pruning_ratios = pruning_ratios
        logger.debug("[DAS] Computed pruning ratios from spectral entropy for {} layers.", len(pruning_ratios))
        return pruning_ratios

    # ------------------------------------------------------------------
    # Activation-size profiling (for TGI memory importance)
    # ------------------------------------------------------------------

    def _get_layer_memories(self) -> tuple[dict[str, float], float]:
        """Return ``{param_name: activation_size}`` and total memory."""
        layer_memories: dict[str, float] = {}
        memory_sum = 0.0
        for name, mod in self.model.named_modules():
            if isinstance(mod, (AutoFreezeConv2d, DASBatchNorm2d, AutoFreezeFC)):
                key = name + ".weight"
                mem = float(getattr(mod, "activation_size", 1))
                layer_memories[key] = mem
                memory_sum += mem
        return layer_memories, memory_sum

    # ------------------------------------------------------------------
    # Sparsity activation / deactivation
    # ------------------------------------------------------------------

    def activate_sparsity(self, pruning_ratios: dict[str, float] | None = None) -> None:
        """Enable DAS on all AutoFreeze modules.

        Parameters
        ----------
        pruning_ratios : dict, optional
            ``{param_name: clip_ratio}``.  If *None*, the last ratios from
            ``probe_and_set_ratios`` are used.
        """
        ratios = pruning_ratios if pruning_ratios is not None else self._pruning_ratios
        for name, mod in self.model.named_modules():
            if isinstance(mod, (AutoFreezeConv2d, DASBatchNorm2d, AutoFreezeFC)):
                mod.sparsity_signal = True
                key = name + ".weight"
                mod.clip_ratio = ratios.get(key, 0.0)

    def deactivate_sparsity(self) -> None:
        """Disable DAS on all AutoFreeze modules (standard training)."""
        for _, mod in self.model.named_modules():
            if isinstance(mod, (AutoFreezeConv2d, DASBatchNorm2d, AutoFreezeFC)):
                mod.sparsity_signal = False

    # ------------------------------------------------------------------
    # Gradient-importance probing → pruning-ratio computation
    # ------------------------------------------------------------------

    @torch.enable_grad()
    def probe_and_set_ratios(
        self,
        x: torch.Tensor,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> dict[str, float]:
        """Probe gradient importance and compute per-layer pruning ratios.

        This is the *Phase-1* step of SURGEON's two-phase approach.  A
        forward-backward pass is executed on a (sub-sampled) batch with
        sparsity disabled.  Gradients are used to compute TGI per layer,
        which is then normalised to derive ``clip_ratio = 1 − TGI /
        max(TGI)``.

        Parameters
        ----------
        x : Tensor
            A batch of input data (intermediate features from edge).
        loss_fn : Callable
            ``loss_fn(logits) → scalar``.  E.g. softmax entropy.

        Returns
        -------
        dict mapping parameter name → pruning ratio ∈ [0, 1].
        """
        self.deactivate_sparsity()
        self.model.train()

        # Sub-sample
        if x.dim() >= 1 and x.shape[0] > self.probe_samples:
            indices = torch.randperm(x.shape[0])[: self.probe_samples]
            x_probe = x[indices]
        else:
            x_probe = x

        # Forward
        logits = self.model(x_probe)
        loss = loss_fn(logits)
        self.model.zero_grad()
        loss.backward()

        # Collect gradients
        layer_names: list[str] = []
        grads: list[torch.Tensor] = []
        params: list[torch.Tensor] = []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                layer_names.append(name)
                grads.append(param.grad.clone())
                params.append(param)

        # Memory importance from activation sizes
        mem_dict, mem_sum = self._get_layer_memories()
        layer_mems = [mem_dict.get(n, 1.0) for n in layer_names]

        # TGI
        tgi = compute_tgi(params, grads, layer_names, layer_mems, mem_sum)

        # Normalise → pruning ratios
        if tgi:
            max_tgi = max(tgi.values()) or 1.0
            pruning_ratios = {k: max(0.0, 1.0 - v / max_tgi) for k, v in tgi.items()}
        else:
            pruning_ratios = {}

        self.model.zero_grad()
        self._pruning_ratios = pruning_ratios

        logger.debug("[DAS] Computed pruning ratios for {} layers.", len(pruning_ratios))
        return pruning_ratios

    @torch.enable_grad()
    def probe_with_targets(
        self,
        forward_fn: Callable[..., dict[str, torch.Tensor]],
        loss_reduce: Callable[[dict[str, torch.Tensor]], torch.Tensor] | None = None,
        **forward_kwargs,
    ) -> dict[str, float]:
        """Probe using an arbitrary forward function (e.g. detection model).

        Unlike ``probe_and_set_ratios`` which assumes a classification-style
        ``model(x) → logits``, this variant accepts a custom *forward_fn*
        that produces a loss dict (typical of detection models).

        Parameters
        ----------
        forward_fn : Callable
            ``forward_fn(**forward_kwargs) → dict[str, Tensor]`` (loss dict).
        loss_reduce : Callable, optional
            Reduce loss dict to a scalar.  Default: ``sum(values())``.
        **forward_kwargs
            Passed directly to *forward_fn*.

        Returns
        -------
        Pruning-ratio dict.
        """
        self.deactivate_sparsity()
        self.model.train()

        loss_dict = forward_fn(**forward_kwargs)

        if loss_reduce is not None:
            loss = loss_reduce(loss_dict)
        else:
            loss = sum(loss_dict.values())

        self.model.zero_grad()
        loss.backward()

        layer_names, grads, params = [], [], []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                layer_names.append(name)
                grads.append(param.grad.clone())
                params.append(param)

        mem_dict, mem_sum = self._get_layer_memories()
        layer_mems = [mem_dict.get(n, 1.0) for n in layer_names]
        tgi = compute_tgi(params, grads, layer_names, layer_mems, mem_sum)

        if tgi:
            max_tgi = max(tgi.values()) or 1.0
            pruning_ratios = {k: max(0.0, 1.0 - v / max_tgi) for k, v in tgi.items()}
        else:
            pruning_ratios = {}

        self.model.zero_grad()
        self._pruning_ratios = pruning_ratios
        logger.debug("[DAS] Computed pruning ratios (probe_with_targets) for {} layers.",
                     len(pruning_ratios))
        return pruning_ratios

    # ------------------------------------------------------------------
    # Convenience: full DAS training step (probe + train)
    # ------------------------------------------------------------------

    def das_train_step(
        self,
        x: torch.Tensor,
        targets: Any,
        loss_fn: Callable[[torch.Tensor, Any], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        *,
        probe_loss_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> float:
        """Complete two-phase DAS training step.

        1. Probe phase: compute pruning ratios (optionally with
           ``probe_loss_fn``; defaults to softmax-entropy).
        2. Train phase: forward-backward-update with activation sparsity.

        Returns the training loss as a float.
        """
        # Phase 1: Probe
        if probe_loss_fn is None:
            def _entropy(logits):
                return -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()
            probe_loss_fn = _entropy

        try:
            self.probe_and_set_ratios(x.detach(), probe_loss_fn)
        except Exception as exc:
            logger.debug("[DAS] Probing failed ({}), using zero pruning.", exc)
            self._pruning_ratios = {}

        # Phase 2: Train with sparsity
        self.activate_sparsity()
        self.model.train()

        output = self.model(x)
        loss = loss_fn(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    @property
    def pruning_ratios(self) -> dict[str, float]:
        """Current per-layer pruning ratios."""
        return dict(self._pruning_ratios)

    def get_memory_stats(self) -> dict:
        """Return activation memory statistics for profiling."""
        total = 0
        cached = 0
        per_layer = {}
        for name, mod in self.model.named_modules():
            if isinstance(mod, (AutoFreezeConv2d, DASBatchNorm2d, AutoFreezeFC)):
                act = getattr(mod, "activation_size", 0)
                bc = getattr(mod, "back_cache_size", 0)
                per_layer[name] = {"activation_size": act, "back_cache_size": bc,
                                   "clip_ratio": getattr(mod, "clip_ratio", 0.0)}
                total += act
                cached += bc
        return {
            "total_activation_elements": total,
            "cached_elements": cached,
            "compression_ratio": cached / max(total, 1),
            "per_layer": per_layer,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 7.  Integration helpers
# ═══════════════════════════════════════════════════════════════════════════

def apply_das_to_model(
    model: nn.Module,
    *,
    bn_only: bool = False,
    probe_samples: int = 10,
    use_spectral_entropy: bool = False,
    device: str | torch.device = "cpu",
) -> DASTrainer:
    """Convenience wrapper — apply DAS to *model* and return a ``DASTrainer``.

    This is the recommended entry point for enabling activation sparsity.
    """
    return DASTrainer(
        model,
        bn_only=bn_only,
        probe_samples=probe_samples,
        use_spectral_entropy=use_spectral_entropy,
        device=device,
    )


def apply_das_to_tail(
    model: nn.Module,
    tail_module_names: list[str],
    *,
    bn_only: bool = False,
    use_spectral_entropy: bool = False,
    device: str | torch.device = "cpu",
) -> DASTrainer:
    """Apply DAS only to specific sub-modules (the *tail* partition).

    Parameters
    ----------
    model : nn.Module
        The full model.
    tail_module_names : list[str]
        Dot-separated module names to DAS-ify, e.g.
        Detector-tail module names such as ``["head"]`` or ``["roi_heads"]``.
    bn_only : bool
    device : device

    Returns
    -------
    DASTrainer (wrapping the full model, but only tail modules are replaced).
    """
    # Temporarily wrap: only replace in the named sub-modules
    trainer = DASTrainer.__new__(DASTrainer)
    trainer.model = model
    trainer.bn_only = bn_only
    trainer.tau = 0.5
    trainer.probe_samples = 10
    trainer.use_spectral_entropy = use_spectral_entropy
    trainer.device = torch.device(device)
    trainer._pruning_ratios = {}

    n_conv = n_bn = n_fc = 0
    for tname in tail_module_names:
        submod = model
        for part in tname.split("."):
            submod = getattr(submod, part, None)
            if submod is None:
                break
        if submod is None:
            logger.warning("[DAS] Tail module '{}' not found, skipping.", tname)
            continue
        n_conv += trainer._replace_conv(submod, tname)
        n_bn += trainer._replace_bn(submod, tname)
        n_fc += trainer._replace_fc(submod, tname)

    trainer._n_conv = n_conv
    trainer._n_bn = n_bn
    trainer._n_fc = n_fc
    logger.info(
        "[DAS] Applied to tail modules {}: {} Conv + {} BN + {} FC replaced.",
        tail_module_names, n_conv, n_bn, n_fc,
    )
    if trainer.use_spectral_entropy:
        trainer.enable_spectral_entropy_tracking(True)
    return trainer
