from __future__ import annotations

import torch

from model_management.split_runtime import (
    fixed_split_runtime_template_key,
    make_split_spec,
    prepare_split_runtime,
)
from model_management.split_runtime.template import (
    FixedSplitRuntimeTemplate,
    FixedSplitRuntimeTemplateCache,
    bind_request_runtime_from_template,
)


class _TwoLayerToyModel(torch.nn.Module):
    def __init__(self, suffix_weight: float):
        super().__init__()
        self.prefix = torch.nn.Linear(1, 1, bias=False)
        self.suffix = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.prefix.weight.fill_(1.0)
            self.suffix.weight.fill_(float(suffix_weight))

    def forward(self, inputs):
        return self.suffix(self.prefix(inputs))


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


def test_runtime_template_bind_uses_request_model_for_suffix_training():
    sample_input = torch.ones(2, 1)
    split_spec = make_split_spec(
        "auto",
        dynamic_batch=(2, 4),
        trainable=True,
        model_family="toy",
    )
    template_model = _TwoLayerToyModel(suffix_weight=2.0)
    request_model = _TwoLayerToyModel(suffix_weight=5.0)
    runtime = prepare_split_runtime(
        template_model,
        sample_input,
        split_spec,
        mode="generated_eager",
    )
    template = FixedSplitRuntimeTemplate(
        cache_key=fixed_split_runtime_template_key(
            model_name="toy",
            model_family="toy",
            split_spec=split_spec,
            example_inputs=sample_input,
            graph_signature=runtime.graph_signature,
        ),
        runtime=runtime,
        split_spec=split_spec,
        model_name="toy",
        model_family="toy",
        graph_signature=runtime.graph_signature,
        symbolic_input_schema_hash="schema",
        split_plan_hash="plan",
    )

    rebound = bind_request_runtime_from_template(template, model=request_model)
    boundary = rebound.run_prefix(sample_input)

    assert rebound.run_suffix(boundary).detach().tolist() == [[5.0], [5.0]]

    optimizer = torch.optim.SGD(request_model.parameters(), lr=0.1)

    def loss_fn(outputs, targets):
        return ((outputs - targets) ** 2).mean()

    loss, _ = rebound.train_suffix(
        boundary,
        torch.zeros(2, 1),
        loss_fn=loss_fn,
        optimizer=optimizer,
    )

    assert loss.item() == 25.0
    assert request_model.suffix.weight.item() < 5.0
    assert template_model.suffix.weight.item() == 2.0
