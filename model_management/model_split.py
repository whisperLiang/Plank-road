from __future__ import annotations

from collections import OrderedDict
from typing import Any

import cv2
import torch


def _frame_to_tensor(frame: Any) -> torch.Tensor:
    if isinstance(frame, torch.Tensor):
        tensor = frame.detach().clone()
        if tensor.dim() == 4:
            if tensor.shape[0] != 1:
                raise ValueError("Only batch size 1 tensors are supported by extract_backbone_features.")
            tensor = tensor[0]
        if tensor.dim() != 3:
            raise ValueError(f"Expected CHW tensor or NCHW tensor, got shape {tuple(tensor.shape)}")
        return tensor.float()

    if frame is None:
        raise ValueError("frame must not be None")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor


def extract_backbone_features(model: torch.nn.Module, frame: Any):
    """Compatibility helper for older scripts that expect backbone features.

    For torchvision detection models, this runs the model's preprocessing
    transform first and then invokes the real backbone on the transformed
    image tensor. For simpler modules that expose a ``backbone`` attribute, it
    falls back to a direct backbone call.
    """

    image = _frame_to_tensor(frame)
    device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu")
    image = image.to(device)

    if hasattr(model, "transform") and hasattr(model, "backbone"):
        model.eval()
        with torch.no_grad():
            image_list, _ = model.transform([image], None)
            return model.backbone(image_list.tensors)

    if hasattr(model, "backbone"):
        model.eval()
        with torch.no_grad():
            return model.backbone(image.unsqueeze(0))

    raise TypeError(f"Model of type {type(model)!r} does not expose a compatible backbone.")


__all__ = ["extract_backbone_features"]
