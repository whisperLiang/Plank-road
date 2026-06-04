"""Lightweight public exports for model management helpers.

Submodules such as ``model_management.payload`` are imported by the cloud
feature-shard layer. Keep package-level exports lazy so those imports do not
eagerly pull in training code and recreate feature-cache import cycles.
"""

from __future__ import annotations

from typing import Any


_UNIVERSAL_EXPORTS = {
    "UniversalModelSplitter",
    "extract_split_features",
    "universal_split_retrain",
    "SplitCandidate",
    "CandidateProfile",
}

_ACTIVATION_SPARSITY_EXPORTS = {
    "DASTrainer",
    "apply_das_to_model",
    "apply_das_to_tail",
    "AutoFreezeConv2d",
    "DASBatchNorm2d",
    "DASGroupNorm",
    "DASLayerNorm",
    "AutoFreezeFC",
    "ActivationClipper",
    "compute_tgi",
}

__all__ = sorted(_UNIVERSAL_EXPORTS | _ACTIVATION_SPARSITY_EXPORTS | {"BoundaryPayload"})


def __getattr__(name: str) -> Any:
    if name in _UNIVERSAL_EXPORTS:
        from model_management import universal_model_split

        value = getattr(universal_model_split, name)
        globals()[name] = value
        return value
    if name in _ACTIVATION_SPARSITY_EXPORTS:
        from model_management import activation_sparsity

        value = getattr(activation_sparsity, name)
        globals()[name] = value
        return value
    if name == "BoundaryPayload":
        from model_management.payload import BoundaryPayload

        globals()[name] = BoundaryPayload
        return BoundaryPayload
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Model-zoo imports pull optional detector runtimes such as torchvision and
# ultralytics. Keep package import lightweight; callers that need model-zoo
# APIs should import model_management.model_zoo directly.
