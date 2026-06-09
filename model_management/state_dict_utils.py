from __future__ import annotations

from collections.abc import Mapping

import torch


def filter_state_dict_to_model_shapes(
    model: torch.nn.Module,
    state_dict: Mapping[str, object],
) -> tuple[dict[str, object], list[str]]:
    model_state = model.state_dict()
    filtered: dict[str, object] = {}
    skipped: list[str] = []
    for key, value in state_dict.items():
        key_str = str(key)
        expected = model_state.get(key_str)
        if (
            expected is not None
            and torch.is_tensor(value)
            and tuple(value.shape) != tuple(expected.shape)
        ):
            skipped.append(key_str)
            continue
        filtered[key_str] = value
    return filtered, skipped
