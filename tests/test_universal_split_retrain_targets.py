from __future__ import annotations

import torch

from model_management.payload import boundary_payload_from_tensors
from model_management.universal_model_split import _load_cached_split_batches


def test_cached_split_batches_attach_coordinate_metadata_to_targets(tmp_path) -> None:
    payload = boundary_payload_from_tensors(
        {"feat": torch.ones(1, 2, 3)},
        split_id="after:feat",
        graph_signature="unit-test",
        batch_size=1,
    )
    record = {
        "intermediate": payload,
        "input_image_size": [1080, 1920],
        "input_tensor_shape": [1, 3, 736, 1280],
        "input_resize_mode": "letterbox",
    }
    annotation = {
        "boxes": [[10.0, 20.0, 30.0, 40.0]],
        "labels": [1],
        "label_coordinate_space": "original_xyxy",
    }

    batches = _load_cached_split_batches(
        cache_path=str(tmp_path),
        all_indices=["sample-1"],
        annotations={"sample-1": annotation},
        batch_size=1,
        runtime=None,
        preloaded_records={"sample-1": record},
    )

    assert len(batches) == 1
    _batch_indices, _boundary, targets = batches[0]
    assert targets[0]["_split_meta"] == {
        "input_image_size": [1080, 1920],
        "input_tensor_shape": [1, 3, 736, 1280],
        "input_resize_mode": "letterbox",
    }
    assert "_split_meta" not in annotation


def test_cached_split_batches_preserve_existing_target_split_meta(tmp_path) -> None:
    payload = boundary_payload_from_tensors(
        {"feat": torch.ones(1, 2, 3)},
        split_id="after:feat",
        graph_signature="unit-test",
        batch_size=1,
    )
    record = {
        "intermediate": payload,
        "input_image_size": [1080, 1920],
        "input_tensor_shape": [1, 3, 736, 1280],
        "input_resize_mode": "letterbox",
    }
    annotation = {
        "boxes": [],
        "labels": [],
        "_split_meta": {
            "input_image_size": [720, 1280],
        },
    }

    batches = _load_cached_split_batches(
        cache_path=str(tmp_path),
        all_indices=["sample-1"],
        annotations={"sample-1": annotation},
        batch_size=1,
        runtime=None,
        preloaded_records={"sample-1": record},
    )

    _batch_indices, _boundary, targets = batches[0]
    assert targets[0]["_split_meta"] == {
        "input_image_size": [720, 1280],
        "input_tensor_shape": [1, 3, 736, 1280],
        "input_resize_mode": "letterbox",
    }
