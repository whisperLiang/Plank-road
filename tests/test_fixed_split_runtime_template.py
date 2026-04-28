from __future__ import annotations

import torch

from model_management.split_runtime import fixed_split_runtime_template_key, make_split_spec
from model_management.split_runtime.template import FixedSplitRuntimeTemplateCache


def test_fixed_split_template_key_symbolizes_batch_size():
    spec = make_split_spec("auto", model_family="yolo")
    key_a = fixed_split_runtime_template_key(
        model_name="toy",
        model_family="yolo",
        split_spec=spec,
        example_inputs=torch.randn(2, 3, 8, 8),
    )
    key_b = fixed_split_runtime_template_key(
        model_name="toy",
        model_family="yolo",
        split_spec=spec,
        example_inputs=torch.randn(3, 3, 8, 8),
    )

    assert key_a == key_b
    assert "trace_batch_size" not in key_a.as_dict()


def test_runtime_template_cache_reuses_same_instance():
    cache = FixedSplitRuntimeTemplateCache()
    key = ("toy", "ariadne")
    calls = {"count": 0}

    def builder():
        calls["count"] += 1
        return object()

    first = cache.get_or_create(key, builder)
    second = cache.get_or_create(key, builder)

    assert first is second
    assert calls["count"] == 1
