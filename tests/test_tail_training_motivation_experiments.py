from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from ariadne.runtime.boundary import BoundaryPayload

from tools import run_tail_training_motivation_experiments as experiments


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
    assert not experiments._is_stable_split_id(
        "after:model.transformer.decoder.layers.0.self_attn"
    )
    assert not experiments._is_stable_split_id(
        "after:model.backbone.0.projector.stages.0.0.m.2.cv2.bn"
    )

    assert (
        experiments._module_level_boundary_for_split_id(
            "after:model.backbone.0.encoder.encoder.encoder.layer.6.mlp.fc2"
        )
        == "after:model.backbone.0.encoder.encoder.encoder.layer.6"
    )
    assert (
        experiments._module_level_boundary_for_split_id(
            "after:model.transformer.decoder.layers.0.self_attn"
        )
        == "after:model.transformer.decoder.layers.0"
    )
    assert (
        experiments._module_level_boundary_for_split_id(
            "after:model.backbone.0.projector.stages.0.0.m.2.cv2.bn"
        )
        == "after:model.backbone.0.projector.stages.0"
    )
    assert experiments._is_stable_split_id(
        "after:model.backbone.0.encoder.encoder.encoder.layer.6"
    )


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


def test_split_position_mode_boxplots_write_pdf_and_png(tmp_path):
    rows = []
    for repeat_id in range(3):
        for bucket, boundary in [
            ("Early25%", "percent:25"),
            ("Middle50%", "percent:50"),
            ("Late75%", "percent:75"),
        ]:
            for mode, time_base, delta_base in [
                ("freeze", 12.0, 0.05),
                ("split_rebuild", 6.0, 0.04),
                ("split_cached", 5.0, 0.03),
            ]:
                rows.append(
                    {
                        "mode": mode,
                        "split_bucket": bucket,
                        "split_boundary": boundary,
                        "repeat_id": repeat_id,
                        "sample_count": 2,
                        "epochs": 1,
                        "train_time_sec": time_base + repeat_id,
                        "metric_delta": delta_base + 0.01 * repeat_id,
                    }
                )

    experiments._write_split_position_mode_boxplots(rows, tmp_path)

    pdf_path = tmp_path / "plots" / "freeze_vs_split_cached_vs_rebuild_by_position.pdf"
    png_path = tmp_path / "plots" / "freeze_vs_split_cached_vs_rebuild_by_position.png"
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0
    assert png_path.exists()
    assert png_path.stat().st_size > 0


class _FakeRuntime:
    split_id = "after:exact"

    def __init__(self):
        self.trained_boundaries = []
        self.fail_lse_once = False

    def train_suffix(self, boundary, targets, *, loss_fn, optimizer):
        del targets, loss_fn, optimizer
        if self.fail_lse_once:
            self.fail_lse_once = False
            raise RuntimeError("LSE is not correctly aligned (strideH)")
        self.trained_boundaries.append(boundary)
        return torch.tensor(0.25), None


def test_split_cached_training_uses_cached_runtime_and_boundary_split_id():
    runtime = _FakeRuntime()
    boundary = SimpleNamespace(split_id=runtime.split_id)
    cached_split = experiments.CachedSplitRuntime(
        percent="percent:50",
        split_id=runtime.split_id,
        runtime=runtime,
        cached_batches=[
            experiments.CachedSplitBatch(
                sample_ids=(1, 2),
                boundary=boundary,
                boundary_split_id=boundary.split_id,
                targets=({"boxes": [], "labels": []}, {"boxes": [], "labels": []}),
            )
        ],
        cache_build_time=1.0,
        runtime_build_time=2.0,
    )

    metrics = experiments._train_split_cached_loop(
        cached_split=cached_split,
        epochs=2,
        loss_fn=lambda _outputs, _targets: torch.tensor(0.25),
        optimizer=None,
        seed=3,
        shuffle_samples=False,
        device=torch.device("cpu"),
    )

    assert runtime.trained_boundaries == [boundary, boundary]
    assert metrics["final_loss"] == pytest.approx(0.25)


def test_split_cached_training_rejects_mismatched_cached_boundary_split_id():
    runtime = _FakeRuntime()
    cached_split = experiments.CachedSplitRuntime(
        percent="percent:75",
        split_id=runtime.split_id,
        runtime=runtime,
        cached_batches=[
            experiments.CachedSplitBatch(
                sample_ids=(7, 8),
                boundary=SimpleNamespace(split_id="after:different"),
                boundary_split_id="after:different",
                targets=(),
            )
        ],
        cache_build_time=1.0,
        runtime_build_time=2.0,
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
    assert "sample index=0" in message
    assert "same SplitPlan" in message


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


def test_split_cached_training_does_not_retry_failed_backward():
    runtime = _FakeRuntime()
    runtime.fail_lse_once = True
    boundary = SimpleNamespace(split_id=runtime.split_id)
    cached_split = experiments.CachedSplitRuntime(
        percent="percent:25",
        split_id=runtime.split_id,
        runtime=runtime,
        cached_batches=[
            experiments.CachedSplitBatch(
                sample_ids=(1, 2),
                boundary=boundary,
                boundary_split_id=boundary.split_id,
                targets=({"boxes": [], "labels": []}, {"boxes": [], "labels": []}),
            )
        ],
        cache_build_time=1.0,
        runtime_build_time=2.0,
    )

    with pytest.raises(RuntimeError, match="LSE is not correctly aligned"):
        experiments._train_split_cached_loop(
            cached_split=cached_split,
            epochs=1,
            loss_fn=lambda _outputs, _targets: torch.tensor(0.25),
            optimizer=None,
            seed=3,
            shuffle_samples=False,
            device=torch.device("cuda"),
        )

    assert runtime.trained_boundaries == []
