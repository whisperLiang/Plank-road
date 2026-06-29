from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from model_management.payload import BoundaryPayload
from model_management.split_runtime import BoundaryPayloadCacheCodec

logger = logging.getLogger(__name__)


class BoundaryFeatureAdapter:
    def __init__(
        self,
        *,
        cosine_weight: float = 1.0,
        nmse_weight: float = 1.0,
        eps: float = 1.0e-8,
        tensor_weights: Mapping[str, float] | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.cosine_weight = float(cosine_weight)
        self.nmse_weight = float(nmse_weight)
        self.eps = float(eps)
        self.tensor_weights = {
            str(key): float(value) for key, value in dict(tensor_weights or {}).items()
        }
        self.device = torch.device(device) if device is not None else None

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | None,
        *,
        device: str | torch.device | None = None,
    ) -> "BoundaryFeatureAdapter":
        cfg = dict(config or {})
        return cls(
            cosine_weight=float(cfg.get("cosine_weight", 1.0)),
            nmse_weight=float(cfg.get("nmse_weight", 1.0)),
            eps=float(cfg.get("eps", 1.0e-8)),
            tensor_weights=cfg.get("tensor_weights") or {},
            device=device,
        )

    def load_payload(self, path: str | Path) -> BoundaryPayload:
        payload = BoundaryPayloadCacheCodec(None).load(path)
        if self.device is None:
            return payload
        tensors = {
            str(label): tensor.to(self.device)
            for label, tensor in dict(payload.tensors).items()
            if isinstance(tensor, torch.Tensor)
        }
        return replace(
            payload, tensors=tensors, spec=dict(payload.spec), metadata=dict(payload.metadata)
        )

    def encode_payload(self, payload: Any) -> dict[str, torch.Tensor]:
        if isinstance(payload, BoundaryPayload):
            source = dict(payload.tensors)
        elif isinstance(payload, torch.Tensor):
            source = {"payload": payload}
        elif isinstance(payload, Mapping):
            raw = payload.get("tensors") if isinstance(payload.get("tensors"), Mapping) else payload
            source = dict(raw)
        else:
            raise TypeError(f"Unsupported boundary payload type: {type(payload).__name__}.")
        tensors: dict[str, torch.Tensor] = {}
        for label, tensor in source.items():
            if not isinstance(tensor, torch.Tensor):
                logger.debug("Skipping non-tensor boundary field %s", label)
                continue
            if not tensor.is_floating_point():
                logger.debug("Skipping non-floating boundary tensor %s", label)
                continue
            tensors[str(label)] = tensor.to(self.device) if self.device is not None else tensor
        return tensors

    def model_forward_to_payload(self, client_model: Any, image_tensor: torch.Tensor) -> Any:
        if hasattr(client_model, "edge_forward"):
            return client_model.edge_forward(image_tensor)
        if hasattr(client_model, "run_prefix"):
            return client_model.run_prefix(image_tensor)
        if callable(client_model):
            return client_model(image_tensor)
        raise TypeError("client_model must expose edge_forward, run_prefix, or be callable.")

    def feature_distance(
        self,
        pred_payload: Any,
        target_payload: Any,
        weights: Mapping[str, float] | None = None,
    ) -> torch.Tensor:
        pred_tensors = self.encode_payload(pred_payload)
        target_tensors = self.encode_payload(target_payload)
        tensor_weights = dict(self.tensor_weights)
        tensor_weights.update(
            {str(key): float(value) for key, value in dict(weights or {}).items()}
        )

        pieces: list[torch.Tensor] = []
        raw_weights: list[float] = []
        for label, pred in pred_tensors.items():
            target = target_tensors.get(label)
            if target is None:
                logger.debug("Skipping boundary tensor %s because target is missing", label)
                continue
            if tuple(pred.shape) != tuple(target.shape):
                logger.debug(
                    "Skipping boundary tensor %s due to shape mismatch pred=%s target=%s",
                    label,
                    tuple(pred.shape),
                    tuple(target.shape),
                )
                continue
            pred_f = pred.float()
            target_f = target.detach().to(device=pred_f.device, dtype=pred_f.dtype)
            if pred_f.numel() == 0 or target_f.numel() == 0:
                logger.debug("Skipping empty boundary tensor %s", label)
                continue

            pred_flat = pred_f.reshape(1, -1)
            target_flat = target_f.reshape(1, -1)
            cosine = 1.0 - F.cosine_similarity(pred_flat, target_flat, dim=1, eps=self.eps).mean()
            numerator = (pred_flat - target_flat).square().sum()
            denominator = target_flat.square().sum().clamp_min(self.eps)
            nmse = numerator / denominator
            pieces.append(self.cosine_weight * cosine + self.nmse_weight * nmse)
            raw_weights.append(max(float(tensor_weights.get(label, 1.0)), 0.0))

        if not pieces:
            raise ValueError("No comparable floating boundary tensors were found.")
        device = pieces[0].device
        weights_t = torch.as_tensor(raw_weights, device=device, dtype=pieces[0].dtype)
        if float(weights_t.sum().detach().cpu()) <= 0.0:
            weights_t = torch.ones_like(weights_t)
        weights_t = weights_t / weights_t.sum().clamp_min(self.eps)
        return sum(weight * piece for weight, piece in zip(weights_t, pieces, strict=True))
