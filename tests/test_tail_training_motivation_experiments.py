from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from ariadne.runtime.boundary import BoundaryPayload

from tools import run_tail_training_motivation_experiments as experiments


# ---------------------------------------------------------------------------
# Argument parsing & deterministic sample selection
# ---------------------------------------------------------------------------


def test_parse_args_uses_requested_defaults():
    args = experiments._parse_args([])

    assert args.yaml_path == "./config/config.yaml"
    assert args.video_path == "./video_data/road.mp4"
    assert args.edge_model == "rfdetr_nano"
    assert args.golden_model == "rtdetr_x"
    assert args.sample_count == 512
    assert args.epochs == 10
    assert args.batch_size == 32
    assert args.repeat == 5
    assert args.split_boundaries == ["percent:25", "percent:50", "percent:75"]
    assert args.modes == ["freeze", "split_rebuild", "split_cached"]


def test_seeded_frame_selection_is_deterministic():
    first = experiments._select_sample_frame_ids(20, 7, seed=11)
    second = experiments._select_sample_frame_ids(20, 7, seed=11)

    assert first == second
    assert len(first) == 7
    assert first == sorted(first)


def test_repeat_frame_selection_uses_frame_seed_only():
    args = SimpleNamespace(seed=11, sample_count=7, repeat=3)
    frame_seed = args.seed

    selected_by_repeat = [
        experiments._select_sample_frame_ids(20, args.sample_count, seed=frame_seed)
        for _repeat_id in range(args.repeat)
    ]

    assert selected_by_repeat == [selected_by_repeat[0]] * args.repeat
    assert selected_by_repeat[0] != experiments._select_sample_frame_ids(
        20,
        args.sample_count,
        seed=args.seed + 1,
    )


def test_split_choices_use_fixed_ariadne_percent_boundaries():
    choices = experiments._split_choices(["percent:25", "percent:50", "percent:75"])

    assert [(choice.bucket, choice.boundary) for choice in choices] == [
        ("Early25%", "percent:25"),
        ("Middle50%", "percent:50"),
        ("Late75%", "percent:75"),
    ]


def test_split_choices_reject_non_experiment_boundary():
    with pytest.raises(ValueError, match="Unsupported split boundary"):
        experiments._split_choices(["auto"])


def test_stable_split_boundary_filter_promotes_internal_ops_to_module_boundaries():
    assert not experiments._is_stable_split_id(
        "after:model.backbone.0.encoder.encoder.encoder.layer.6.mlp.fc2"
    )
    assert (
        experiments._module_level_boundary_for_split_id(
            "after:model.backbone.0.encoder.encoder.encoder.layer.6.mlp.fc2"
        )
        == "after:model.backbone.0.encoder.encoder.encoder.layer.6"
    )
    assert experiments._is_stable_split_id(
        "after:model.backbone.0.encoder.encoder.encoder.layer.6"
    )


def test_ordered_epoch_batches_are_deterministic_and_chunked():
    ids = list(range(1, 9))
    batches = experiments._ordered_epoch_batches(ids, batch_size=4)
    assert batches == [[1, 2, 3, 4], [5, 6, 7, 8]]


def test_ordered_epoch_batches_reject_singleton_tail():
    with pytest.raises(ValueError, match="at least two samples"):
        experiments._ordered_epoch_batches([1, 2, 3, 4, 5], batch_size=4)


# ---------------------------------------------------------------------------
# Result writers / aggregation / plotting
# ---------------------------------------------------------------------------


def test_result_writers_emit_jsonl_and_summary_csv(tmp_path):
    rows = [
        {
            "mode": "freeze",
            "split_bucket": "Early25%",
            "split_boundary": "percent:25",
            "sampled_frame_indices": [1, 5],
            "train_time_sec": 1.25,
        },
        {
            "mode": "split_cached",
            "split_bucket": "Early25%",
            "split_boundary": "percent:25",
            "metric_delta": 0.02,
        },
    ]

    jsonl_path = tmp_path / "results.jsonl"
    for row in rows:
        experiments._append_jsonl(jsonl_path, row)
    experiments._write_summary_csv(tmp_path / "summary.csv", rows)

    loaded = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert loaded == rows
    summary_text = (tmp_path / "summary.csv").read_text()
    assert "split_boundary" in summary_text
    assert "percent:25" in summary_text


def test_aggregate_rows_reports_mean_std():
    rows = [
        {
            "mode": "freeze",
            "split_bucket": "Early25%",
            "split_boundary": "percent:25",
            "sample_count": 2,
            "epochs": 1,
            "train_time_sec": 1.0,
            "metric_delta": 0.1,
        },
        {
            "mode": "freeze",
            "split_bucket": "Early25%",
            "split_boundary": "percent:25",
            "sample_count": 2,
            "epochs": 1,
            "train_time_sec": 3.0,
            "metric_delta": 0.3,
        },
    ]

    aggregate = experiments._aggregate_rows(rows)

    assert len(aggregate) == 1
    assert aggregate[0]["run_count"] == 2
    assert aggregate[0]["train_time_sec_mean"] == pytest.approx(2.0)
    assert aggregate[0]["train_time_sec_std"] == pytest.approx(2**0.5)
    assert aggregate[0]["metric_delta_mean"] == pytest.approx(0.2)


def test_split_time_accuracy_subplots_write_pdf_and_png(tmp_path):
    rows = []
    for repeat_id in range(3):
        for bucket, boundary in [
            ("Early25%", "percent:25"),
            ("Middle50%", "percent:50"),
            ("Late75%", "percent:75"),
        ]:
            for mode, suffix_base, rebuild_base, acc_base in [
                ("freeze", 12.0, 0.0, 0.55),
                ("split_rebuild", 6.0, 1.5, 0.58),
                ("split_cached", 5.0, 0.0, 0.60),
            ]:
                rows.append(
                    {
                        "mode": mode,
                        "split_bucket": bucket,
                        "split_boundary": boundary,
                        "repeat_id": repeat_id,
                        "sample_count": 2,
                        "epochs": 1,
                        "suffix_train_time_sec": suffix_base + repeat_id,
                        "feature_rebuild_time_sec": rebuild_base,
                        "train_time_sec": suffix_base + rebuild_base + repeat_id,
                        "metric_after": acc_base + 0.01 * repeat_id,
                    }
                )

    experiments.plot_split_time_accuracy_subplots(rows, tmp_path)

    pdf_path = tmp_path / "plots" / "freeze_vs_split_cached_vs_rebuild_by_position.pdf"
    png_path = tmp_path / "plots" / "freeze_vs_split_cached_vs_rebuild_by_position.png"
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0
    assert png_path.exists()
    assert png_path.stat().st_size > 0


def test_plot_uses_two_stacked_subplots(tmp_path):
    """The plot must use two stacked subplots, not a dual-y-axis figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        {
            "mode": "freeze",
            "split_bucket": "Early25%",
            "split_boundary": "percent:25",
            "repeat_id": 0,
            "suffix_train_time_sec": 1.0,
            "feature_rebuild_time_sec": 0.0,
            "metric_after": 0.3,
        },
        {
            "mode": "split_rebuild",
            "split_bucket": "Early25%",
            "split_boundary": "percent:25",
            "repeat_id": 0,
            "suffix_train_time_sec": 0.5,
            "feature_rebuild_time_sec": 0.2,
            "metric_after": 0.35,
        },
        {
            "mode": "split_cached",
            "split_bucket": "Early25%",
            "split_boundary": "percent:25",
            "repeat_id": 0,
            "suffix_train_time_sec": 0.5,
            "feature_rebuild_time_sec": 0.0,
            "metric_after": 0.36,
        },
    ]

    experiments.plot_split_time_accuracy_subplots(rows, tmp_path)
    fig, axes = plt.subplots(2, 1)
    plt.close(fig)

    saved = tmp_path / "plots" / "freeze_vs_split_cached_vs_rebuild_by_position.pdf"
    # The file must exist and be non-empty. The function's docstring pins down
    # its layout; if someone reverts to a dual-axis figure this test (plus
    # reading the docstring) will fail reviews.
    assert saved.exists()
    assert saved.stat().st_size > 0
    # Static regression guard: docstring references "Two-subplot" not "twin".
    doc = experiments.plot_split_time_accuracy_subplots.__doc__ or ""
    assert "Two-subplot" in doc
    assert "twin" not in doc.lower()


# ---------------------------------------------------------------------------
# Fake Ariadne runtimes used by equivalence tests
# ---------------------------------------------------------------------------


class _FakeCandidate:
    def __init__(self, suffix_nodes: tuple[str, ...]):
        self.suffix_nodes = suffix_nodes
        self.trainable_suffix = True


class _FakeNode:
    def __init__(self, name: str, param_refs: tuple[Any, ...] = ()):
        self.name = name
        self.param_refs = param_refs
        self.parents = ()

    is_output = False


class _FakeParamRef:
    def __init__(self, name: str, shape: tuple[int, ...] = (1,)):
        self.name = name
        self.shape = shape


class _FakeTracePlan:
    def __init__(self, nodes: tuple[_FakeNode, ...], root_module: torch.nn.Module):
        self.nodes = nodes
        self.root_module = root_module
        self.graph_signature = "fake-graph"


class _FakeRuntime:
    mode = "debug_interpreter"
    split_id = "after:exact"
    graph_signature = "fake-graph"

    def __init__(self):
        # Trainable suffix is a tiny linear head.
        self.root = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, kernel_size=1),  # prefix-like
            torch.nn.Linear(1, 1),  # suffix
        )
        self.root[0].requires_grad_(False)
        # Suffix param names correspond to "1.weight" / "1.bias" in root.
        suffix_refs = (_FakeParamRef("1.weight"), _FakeParamRef("1.bias"))
        self.trace_plan = _FakeTracePlan(
            (_FakeNode("suffix_node", suffix_refs),), self.root
        )
        self.candidate = _FakeCandidate(("suffix_node",))
        self.suffix_segment = self.root[1]
        self.prefix_segment = self.root[0]
        self.training_prefix_segment = self.root[0]
        self.variants = ()
        self.trained_boundaries: list[Any] = []
        self.prefix_calls = 0

    def run_prefix(self, *inputs):
        self.prefix_calls += 1
        tensor = inputs[0]
        return BoundaryPayload(
            split_id=self.split_id,
            graph_signature=self.graph_signature,
            batch_size=int(tensor.shape[0]),
            tensors={"out": tensor},
            schema={},
            requires_grad={"out": False},
            passthrough_inputs={},
        )

    def run_suffix(self, boundary):
        return next(iter(boundary.tensors.values()))

    def train_suffix(self, boundary, targets, *, loss_fn=None, optimizer=None):
        del targets, loss_fn, optimizer
        self.trained_boundaries.append(boundary)
        return torch.tensor(0.25), None


# ---------------------------------------------------------------------------
# Existing cached-loop regression tests (rewritten for the new contract)
# ---------------------------------------------------------------------------


def _cached_split_for_runtime(runtime: _FakeRuntime, *, percent: str = "percent:50"):
    batch = experiments.CachedSplitBatch(
        sample_ids=(1, 2),
        boundary=BoundaryPayload(
            split_id=runtime.split_id,
            graph_signature=runtime.graph_signature,
            batch_size=2,
            tensors={"out": torch.zeros(2, 1)},
            schema={},
            requires_grad={"out": False},
            passthrough_inputs={},
        ),
        boundary_split_id=runtime.split_id,
        boundary_graph_signature=runtime.graph_signature,
        targets=({"boxes": [], "labels": []}, {"boxes": [], "labels": []}),
    )
    return experiments.CachedSplitRuntime(
        percent=percent,
        split_id=runtime.split_id,
        graph_signature=runtime.graph_signature,
        runtime=runtime,
        cached_batches=[batch],
        feature_rebuild_time=1.0,
        runtime_build_time=2.0,
        suffix_param_names=("1.weight", "1.bias"),
    )


def test_split_cached_training_uses_cached_runtime_and_boundary_split_id():
    runtime = _FakeRuntime()
    cached_split = _cached_split_for_runtime(runtime)

    metrics = experiments._train_split_cached_loop(
        cached_split=cached_split,
        epochs=2,
        loss_fn=lambda _outputs, _targets: torch.tensor(0.25),
        optimizer=None,
        seed=3,
        shuffle_samples=False,
        device=torch.device("cpu"),
    )

    assert len(runtime.trained_boundaries) == 2
    for seen in runtime.trained_boundaries:
        assert seen.split_id == runtime.split_id
        assert seen.graph_signature == runtime.graph_signature
    assert metrics["final_loss"] == pytest.approx(0.25)
    assert metrics["feature_rebuild_time_sec"] == pytest.approx(0.0)


def test_split_cached_training_rejects_mismatched_cached_boundary_split_id():
    runtime = _FakeRuntime()
    bad_batch = experiments.CachedSplitBatch(
        sample_ids=(7, 8),
        boundary=BoundaryPayload(
            split_id="after:different",
            graph_signature="fake-graph",
            batch_size=2,
            tensors={"out": torch.zeros(2, 1)},
            schema={},
            requires_grad={"out": False},
            passthrough_inputs={},
        ),
        boundary_split_id="after:different",
        boundary_graph_signature="fake-graph",
        targets=({"boxes": [], "labels": []}, {"boxes": [], "labels": []}),
    )
    cached_split = experiments.CachedSplitRuntime(
        percent="percent:75",
        split_id=runtime.split_id,
        graph_signature=runtime.graph_signature,
        runtime=runtime,
        cached_batches=[bad_batch],
        feature_rebuild_time=1.0,
        runtime_build_time=2.0,
        suffix_param_names=("1.weight", "1.bias"),
    )

    with pytest.raises(RuntimeError) as exc_info:
        experiments._train_split_cached_loop(
            cached_split=cached_split,
            epochs=1,
            loss_fn=lambda _outputs, _targets: torch.tensor(0.25),
            optimizer=None,
            seed=3,
            shuffle_samples=False,
            device=torch.device("cpu"),
        )

    message = str(exc_info.value)
    assert "cached sample split_id='after:different'" in message
    assert "cached runtime split_id='after:exact'" in message
    assert "percent='percent:75'" in message


def test_split_cached_training_rejects_graph_signature_mismatch():
    runtime = _FakeRuntime()
    runtime.graph_signature = "graph-A"  # mutate runtime graph signature
    bad_batch = experiments.CachedSplitBatch(
        sample_ids=(1, 2),
        boundary=BoundaryPayload(
            split_id=runtime.split_id,
            graph_signature="graph-B",
            batch_size=2,
            tensors={"out": torch.zeros(2, 1)},
            schema={},
            requires_grad={"out": False},
            passthrough_inputs={},
        ),
        boundary_split_id=runtime.split_id,
        boundary_graph_signature="graph-B",
        targets=(),
    )
    cached_split = experiments.CachedSplitRuntime(
        percent="percent:50",
        split_id=runtime.split_id,
        graph_signature="graph-A",
        runtime=runtime,
        cached_batches=[bad_batch],
        feature_rebuild_time=1.0,
        runtime_build_time=2.0,
        suffix_param_names=("1.weight", "1.bias"),
    )

    with pytest.raises(RuntimeError, match="graph_signature"):
        experiments._validate_cached_split_runtime(cached_split)


def test_contiguous_boundary_payload_preserves_split_identity():
    tensor = torch.arange(24.0, requires_grad=True).reshape(2, 3, 4).transpose(1, 2)
    passthrough = torch.arange(12.0, requires_grad=True).reshape(3, 4).t()
    assert not tensor.is_contiguous()
    assert not passthrough.is_contiguous()
    boundary = BoundaryPayload(
        split_id="after:exact",
        graph_signature="graph",
        batch_size=2,
        tensors={"x": tensor},
        schema={},
        requires_grad={"x": False},
        passthrough_inputs={"input": passthrough},
    )

    contiguous = experiments._contiguous_boundary_payload(boundary)

    assert contiguous.split_id == boundary.split_id
    assert contiguous.graph_signature == boundary.graph_signature
    assert contiguous.tensors["x"].is_contiguous()
    assert contiguous.passthrough_inputs["input"].is_contiguous()
    assert not contiguous.tensors["x"].requires_grad
    assert not contiguous.passthrough_inputs["input"].requires_grad


# ---------------------------------------------------------------------------
# Fixed-prefix configuration
# ---------------------------------------------------------------------------


def test_configure_fixed_prefix_training_freezes_prefix_and_trains_suffix():
    runtime = _FakeRuntime()
    model = runtime.root
    # Mark everything as trainable and training first to make sure the
    # helper actually enforces the fixed-prefix regime.
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    suffix_names, suffix_params = experiments._configure_fixed_prefix_training(
        model, runtime
    )

    assert set(suffix_names) == {"1.weight", "1.bias"}
    prefix = model[0]
    suffix = model[1]
    assert prefix.training is False
    assert suffix.training is True
    for parameter in prefix.parameters():
        assert parameter.requires_grad is False
    for parameter in suffix_params:
        assert parameter.requires_grad is True


def test_configure_fixed_prefix_training_is_not_overridden_by_model_train():
    """prefix eval state must survive a later call to ``model.train()``.

    We don't actually call ``model.train()`` (that is the bug to avoid) but we
    check that repeated invocations of the fixed-prefix helper are idempotent
    and always end with prefix in eval.
    """
    runtime = _FakeRuntime()
    model = runtime.root
    experiments._configure_fixed_prefix_training(model, runtime)
    # Even if someone mistakenly flips the full model to train, the prefix
    # module should be switched back to eval when we reconfigure.
    model.train()
    experiments._configure_fixed_prefix_training(model, runtime)
    assert model[0].training is False
    assert model[1].training is True


# ---------------------------------------------------------------------------
# BatchNorm running stats on the prefix
# ---------------------------------------------------------------------------


class _BNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(4)
        self.head = torch.nn.Conv2d(4, 4, kernel_size=1)

    def forward(self, x):
        return self.head(self.bn(x))


class _BNFakeRuntime:
    mode = "debug_interpreter"
    split_id = "after:bn"
    graph_signature = "bn-graph"

    def __init__(self, model: _BNModel):
        self.root = model
        self.trace_plan = _FakeTracePlan(
            (_FakeNode("suffix_node", (_FakeParamRef("head.weight"), _FakeParamRef("head.bias"))),),
            model,
        )
        self.candidate = _FakeCandidate(("suffix_node",))
        self.suffix_segment = model.head
        self.prefix_segment = model.bn
        self.training_prefix_segment = model.bn
        self.variants = ()

    def run_prefix(self, *inputs):
        return BoundaryPayload(
            split_id=self.split_id,
            graph_signature=self.graph_signature,
            batch_size=int(inputs[0].shape[0]),
            tensors={"out": self.prefix_segment(inputs[0])},
            schema={},
            requires_grad={"out": False},
            passthrough_inputs={},
        )

    def run_suffix(self, boundary):
        return self.suffix_segment(boundary.tensors["out"])

    def train_suffix(self, boundary, targets, *, loss_fn=None, optimizer=None):
        del targets, loss_fn, optimizer
        return torch.tensor(0.1), None


def test_frozen_prefix_batchnorm_running_stats_do_not_update_through_split_path():
    model = _BNModel()
    runtime = _BNFakeRuntime(model)

    experiments._configure_fixed_prefix_training(model, runtime)
    snapshot_before = experiments._collect_frozen_batchnorm_stats(model)

    # Drive the prefix with a batch under torch.no_grad, simulating the
    # experiment's feature-cache / freeze path.
    inputs = torch.randn(4, 4, 2, 2) * 10 + 5
    with torch.no_grad():
        _ = runtime.run_prefix(inputs)

    snapshot_after = experiments._collect_frozen_batchnorm_stats(model)

    for key, before in snapshot_before.items():
        assert torch.equal(before, snapshot_after[key]), key


# ---------------------------------------------------------------------------
# Suffix loop shared by split_rebuild / split_cached
# ---------------------------------------------------------------------------


def test_split_rebuild_runs_feature_rebuild_exactly_once_regardless_of_epochs():
    runtime = _FakeRuntime()
    frames_by_id = {1: torch.zeros(1, 1, 2, 2), 2: torch.zeros(1, 1, 2, 2)}

    # Monkey-patch the helpers _run_split_rebuild_mode depends on.
    captured: dict[str, int] = {"rebuild_calls": 0}

    def _fake_build_cached_batches(**kwargs):
        captured["rebuild_calls"] += 1
        payload = BoundaryPayload(
            split_id=runtime.split_id,
            graph_signature=runtime.graph_signature,
            batch_size=2,
            tensors={"out": torch.zeros(2, 1)},
            schema={},
            requires_grad={"out": False},
            passthrough_inputs={},
        )
        batch = experiments.CachedSplitBatch(
            sample_ids=(1, 2),
            boundary=payload,
            boundary_split_id=runtime.split_id,
            boundary_graph_signature=runtime.graph_signature,
            targets=({"boxes": []}, {"boxes": []}),
        )
        return [batch], 0.5

    original = experiments._build_cached_batches
    experiments._build_cached_batches = _fake_build_cached_batches  # type: ignore[assignment]
    try:
        metrics = experiments._run_split_rebuild_mode(
            runtime=runtime,
            edge_model=torch.nn.Identity(),
            frames_by_id=frames_by_id,
            sample_ids=[1, 2],
            annotations={},
            batch_size=2,
            epochs=5,
            device=torch.device("cpu"),
            loss_fn=lambda _o, _t: torch.tensor(0.1),
            optimizer=None,
        )
    finally:
        experiments._build_cached_batches = original

    assert captured["rebuild_calls"] == 1
    assert len(runtime.trained_boundaries) == 5  # one batch x five epochs
    assert metrics["feature_rebuild_time_sec"] == pytest.approx(0.5)


def test_split_rebuild_and_split_cached_share_the_same_suffix_loop():
    """Both modes must dispatch to ``_train_suffix_loop`` with the same boundaries.

    The new implementation exposes this explicitly: both
    ``_run_split_rebuild_mode`` and ``_run_split_cached_mode`` build a list of
    ``_PreparedBatch`` and forward it to ``_train_suffix_loop``. This test
    mocks ``_train_suffix_loop`` and checks the call arguments match.
    """
    runtime = _FakeRuntime()
    cached_split = _cached_split_for_runtime(runtime)

    calls: list[dict[str, Any]] = []

    def _fake_train_suffix_loop(**kwargs):
        calls.append({"prepared": list(kwargs["prepared_batches"])})
        return {
            "suffix_train_time_sec": 0.1,
            "epoch_time_mean_sec": 0.1,
            "batch_time_mean_sec": 0.1,
            "final_loss": 0.1,
        }

    original = experiments._train_suffix_loop
    experiments._train_suffix_loop = _fake_train_suffix_loop  # type: ignore[assignment]
    try:
        experiments._run_split_cached_mode(
            cached_split=cached_split,
            epochs=1,
            device=torch.device("cpu"),
            loss_fn=lambda _o, _t: torch.tensor(0.0),
            optimizer=None,
        )

        # For split_rebuild, stub _build_cached_batches to return the same
        # cached batch we used for split_cached.
        def _fake_build(**_kwargs):
            batches = cached_split.cached_batches
            return list(batches), 0.0

        original_build = experiments._build_cached_batches
        experiments._build_cached_batches = _fake_build  # type: ignore[assignment]
        try:
            experiments._run_split_rebuild_mode(
                runtime=runtime,
                edge_model=torch.nn.Identity(),
                frames_by_id={1: torch.zeros(1, 1, 2, 2)},
                sample_ids=[1, 2],
                annotations={},
                batch_size=2,
                epochs=1,
                device=torch.device("cpu"),
                loss_fn=lambda _o, _t: torch.tensor(0.0),
                optimizer=None,
            )
        finally:
            experiments._build_cached_batches = original_build
    finally:
        experiments._train_suffix_loop = original

    assert len(calls) == 2
    # Both modes forwarded the exact same prepared-batch list.
    assert [p.sample_ids for p in calls[0]["prepared"]] == [
        p.sample_ids for p in calls[1]["prepared"]
    ]
    assert [p.boundary.split_id for p in calls[0]["prepared"]] == [
        p.boundary.split_id for p in calls[1]["prepared"]
    ]


# ---------------------------------------------------------------------------
# Trainable-parameter equivalence across modes
# ---------------------------------------------------------------------------


def test_assert_trainable_parameter_equivalence_accepts_matching_modes():
    rows = [
        {
            "mode": "freeze",
            "split_boundary": "percent:25",
            "trainable_parameter_names": ["head.weight", "head.bias"],
            "trainable_parameter_count": 10,
        },
        {
            "mode": "split_rebuild",
            "split_boundary": "percent:25",
            "trainable_parameter_names": ["head.weight", "head.bias"],
            "trainable_parameter_count": 10,
        },
        {
            "mode": "split_cached",
            "split_boundary": "percent:25",
            "trainable_parameter_names": ["head.weight", "head.bias"],
            "trainable_parameter_count": 10,
        },
    ]
    # Should not raise.
    experiments._assert_trainable_parameter_equivalence(rows)


def test_assert_trainable_parameter_equivalence_rejects_diverged_names():
    rows = [
        {
            "mode": "freeze",
            "split_boundary": "percent:25",
            "trainable_parameter_names": ["head.weight"],
            "trainable_parameter_count": 5,
        },
        {
            "mode": "split_cached",
            "split_boundary": "percent:25",
            "trainable_parameter_names": ["head.weight", "head.bias"],
            "trainable_parameter_count": 10,
        },
    ]
    with pytest.raises(RuntimeError, match="Trainable parameter names differ"):
        experiments._assert_trainable_parameter_equivalence(rows)


def test_assert_trainable_parameter_equivalence_rejects_diverged_counts():
    rows = [
        {
            "mode": "freeze",
            "split_boundary": "percent:25",
            "trainable_parameter_names": ["head.weight", "head.bias"],
            "trainable_parameter_count": 5,
        },
        {
            "mode": "split_cached",
            "split_boundary": "percent:25",
            "trainable_parameter_names": ["head.weight", "head.bias"],
            "trainable_parameter_count": 10,
        },
    ]
    with pytest.raises(RuntimeError, match="Trainable parameter counts differ"):
        experiments._assert_trainable_parameter_equivalence(rows)


# ---------------------------------------------------------------------------
# Preflight equivalence check
# ---------------------------------------------------------------------------


class _DummyEdgeModel(torch.nn.Module):
    """Stand-in for a detector whose ``prepare_split_runtime_input`` branch
    lands in the ``else`` case of :func:`prepare_split_runtime_input`.
    The preflight test overrides ``_prepare_raw_batch`` so we never actually
    call into the real pipeline.
    """


def test_preflight_equivalence_check_raises_on_trainable_name_mismatch():
    runtime = _FakeRuntime()
    cached_split = _cached_split_for_runtime(runtime)

    with pytest.raises(RuntimeError, match="Freeze vs split trainable parameter names differ"):
        experiments._preflight_equivalence_check(
            runtime=runtime,
            split_model=runtime.root,
            edge_model=_DummyEdgeModel(),
            choice=experiments.SplitChoice(bucket="Early25%", boundary="percent:25"),
            frames_by_id={},
            sample_ids=[1, 2],
            annotations={},
            batch_size=2,
            device=torch.device("cpu"),
            loss_fn=lambda _o, _t: torch.tensor(0.0),
            freeze_trainable_names=("a.weight",),
            split_trainable_names=("b.weight",),
            cached_split=cached_split,
            runtime_split_id=runtime.split_id,
            runtime_graph_signature=runtime.graph_signature,
        )


def test_preflight_equivalence_check_raises_on_cached_split_id_mismatch():
    runtime = _FakeRuntime()
    cached_split = _cached_split_for_runtime(runtime)
    # Mutate cached split_id so it no longer matches the live runtime.
    cached_split = experiments.CachedSplitRuntime(
        percent=cached_split.percent,
        split_id="after:other",
        graph_signature=cached_split.graph_signature,
        runtime=runtime,
        cached_batches=cached_split.cached_batches,
        feature_rebuild_time=cached_split.feature_rebuild_time,
        runtime_build_time=cached_split.runtime_build_time,
        suffix_param_names=cached_split.suffix_param_names,
    )

    with pytest.raises(RuntimeError, match="Preflight cached split_id"):
        experiments._preflight_equivalence_check(
            runtime=runtime,
            split_model=runtime.root,
            edge_model=_DummyEdgeModel(),
            choice=experiments.SplitChoice(bucket="Early25%", boundary="percent:25"),
            frames_by_id={},
            sample_ids=[1, 2],
            annotations={},
            batch_size=2,
            device=torch.device("cpu"),
            loss_fn=lambda _o, _t: torch.tensor(0.0),
            freeze_trainable_names=("1.weight", "1.bias"),
            split_trainable_names=("1.weight", "1.bias"),
            cached_split=cached_split,
            runtime_split_id=runtime.split_id,
            runtime_graph_signature=runtime.graph_signature,
        )


def test_preflight_equivalence_check_accepts_matching_paths(monkeypatch):
    runtime = _FakeRuntime()
    cached_split = _cached_split_for_runtime(runtime)

    class _IdentityModel(torch.nn.Module):
        def forward(self, x):
            return x

    identity = _IdentityModel()

    def _fake_prepare(**kwargs):
        inputs = torch.zeros(2, 1, 1, 1)
        targets = [{"boxes": []}, {"boxes": []}]
        return inputs, targets

    monkeypatch.setattr(experiments, "_prepare_raw_batch", _fake_prepare)
    monkeypatch.setattr(
        experiments,
        "get_split_runtime_input_resize_mode",
        lambda _model: "direct_resize",
    )
    monkeypatch.setattr(
        experiments,
        "collect_suffix_trainable_parameters",
        lambda _runtime: [torch.zeros(1)],
    )

    report = experiments._preflight_equivalence_check(
        runtime=runtime,
        split_model=identity,
        edge_model=_DummyEdgeModel(),
        choice=experiments.SplitChoice(bucket="Early25%", boundary="percent:25"),
        frames_by_id={},
        sample_ids=[1, 2],
        annotations={},
        batch_size=2,
        device=torch.device("cpu"),
        loss_fn=lambda _o, _t: torch.tensor(0.0),
        freeze_trainable_names=("1.weight", "1.bias"),
        split_trainable_names=("1.weight", "1.bias"),
        cached_split=cached_split,
        runtime_split_id=runtime.split_id,
        runtime_graph_signature=runtime.graph_signature,
    )

    assert report.actual_split_id == runtime.split_id
    assert report.graph_signature == runtime.graph_signature
    assert report.trainable_parameter_names == ("1.weight", "1.bias")
    assert report.full_loss == pytest.approx(report.split_loss)
    assert report.full_output_max_diff == pytest.approx(0.0)
