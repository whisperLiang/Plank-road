from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


class ToyDetector(torch.nn.Module):
    model_family = "generic"

    def __init__(self, *, output_kind: str) -> None:
        super().__init__()
        self.output_kind = output_kind
        self.stem = torch.nn.Conv2d(3, 4, kernel_size=1)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.cls = torch.nn.Linear(4, 6)
        self.box = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> Any:
        z = self.pool(torch.relu(self.stem(x))).flatten(1)
        logits = self.cls(z).unsqueeze(1)
        boxes = self.box(z).sigmoid().unsqueeze(1)
        if self.output_kind == "yolo":
            return torch.cat([boxes.flatten(1), logits.flatten(1)], dim=1)
        if self.output_kind == "rfdetr":
            return {"pred_logits": logits, "pred_boxes": boxes}
        if self.output_kind == "tinynext":
            return {"cls_logits": logits, "bbox_regression": boxes}
        raise AssertionError(self.output_kind)


def toy_loss(outputs: Any, targets: Any) -> torch.Tensor:
    del targets
    tensors: list[torch.Tensor] = []

    def walk(value: Any) -> None:
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            tensors.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(outputs)
    assert tensors
    return sum(tensor.float().square().mean() for tensor in tensors) / len(tensors)


def assert_valid_detector_output(output_kind: str, outputs: Any, batch_size: int) -> None:
    if output_kind == "yolo":
        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape[0] == batch_size
        return
    if output_kind == "rfdetr":
        assert set(outputs) >= {"pred_logits", "pred_boxes"}
        assert outputs["pred_logits"].shape[0] == batch_size
        assert outputs["pred_boxes"].shape[0] == batch_size
        return
    if output_kind == "tinynext":
        assert set(outputs) >= {"cls_logits", "bbox_regression"}
        assert outputs["cls_logits"].shape[0] == batch_size
        assert outputs["bbox_regression"].shape[0] == batch_size
        return
    raise AssertionError(output_kind)


@dataclass
class CountingRuntime:
    fail_replay: bool = False
    fail_train: bool = False
    suffix_calls: int = 0
    train_calls: int = 0

    def validate_boundary(self, boundary) -> None:
        return None

    def run_suffix(self, boundary):
        self.suffix_calls += 1
        if self.fail_replay:
            raise RuntimeError("boom")
        return next(iter(boundary.tensors.values()))

    def train_suffix(self, boundary, targets, *, loss_fn=None, optimizer=None):
        del targets, loss_fn, optimizer
        self.train_calls += 1
        if self.fail_train:
            raise RuntimeError("boom")
        tensor = next(iter(boundary.tensors.values()))
        return tensor.sum(), {}
