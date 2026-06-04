from __future__ import annotations

from typing import Any

import torch

from model_management.payload import boundary_payload_from_tensors


def runtime_context(layout_id: str = "layout-a") -> dict[str, Any]:
    return {
        "model_id": "yolo26n",
        "model_family": "yolo",
        "split_config_id": "split-a",
        "contract_id": "contract-a",
        "feature_layout_id": layout_id,
        "boundary_id": "after:test",
        "input_tensor_shape": [1, 3, 320, 320],
        "input_resize_mode": "direct_resize",
    }


def make_entries(count: int, *, dtype: torch.dtype = torch.float16) -> list[dict[str, Any]]:
    entries = []
    for index in range(count):
        payload = boundary_payload_from_tensors(
            {
                "boundary": torch.full((1, 2, 3), float(index), dtype=dtype),
                "skip": torch.full((1, 1, 2), float(index + 10), dtype=dtype),
            },
            split_id="after:test",
            graph_signature="test-graph",
            batch_size=1,
        )
        entries.append(
            {
                "sample": {
                    "sample_id": f"sample-{index}",
                    "sample_source": "high_quality",
                    "labels": {
                        "boxes": [[0.0, 0.0, 1.0, 1.0]],
                        "labels": [1],
                        "label_coordinate_space": "original_xyxy",
                    },
                    "input_tensor_shape": [1, 3, 320, 320],
                    "input_resize_mode": "direct_resize",
                    "input_image_size": [320, 320],
                },
                "record": {"intermediate": payload},
            }
        )
    return entries
