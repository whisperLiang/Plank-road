from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


MODEL_DELTA_PAYLOAD_FORMAT = "state_dict_delta.v1"


def build_state_dict_delta_payload(
    model: torch.nn.Module,
    *,
    model_name: str,
    base_model_version: str,
    result_model_version: str,
) -> dict[str, Any]:
    state = model.state_dict()
    trainable_names = {
        name
        for name, parameter in model.named_parameters()
        if bool(getattr(parameter, "requires_grad", False))
    }
    threshold_names = {
        "plank_threshold_low",
        "plank_threshold_high",
    }
    selected_names = trainable_names | {name for name in state if name in threshold_names}
    return {
        "format": MODEL_DELTA_PAYLOAD_FORMAT,
        "model_name": str(model_name),
        "base_model_version": str(base_model_version),
        "result_model_version": str(result_model_version),
        "state_dict": {
            name: value.detach().cpu() if torch.is_tensor(value) else value
            for name, value in state.items()
            if name in selected_names
        },
    }


def require_state_dict_delta_payload(payload: object) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise RuntimeError("Cloud model update must be a state_dict_delta.v1 payload.")
    if payload.get("format") != MODEL_DELTA_PAYLOAD_FORMAT:
        raise RuntimeError(
            "Unsupported cloud model update format: "
            f"{payload.get('format', '<missing>')!r}; expected {MODEL_DELTA_PAYLOAD_FORMAT!r}."
        )
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise RuntimeError("Cloud model update payload is missing a non-empty state_dict delta.")
    return payload
