from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from tools import run_tail_training_motivation_experiments as experiments


def _candidate(candidate_id: str, ratio: float) -> experiments.SplitCandidate:
    total = 1000
    prefix = int(total * ratio)
    return experiments.SplitCandidate(
        candidate_id=candidate_id,
        edge_nodes=[],
        cloud_nodes=[],
        boundary_edges=[],
        boundary_tensor_labels=[candidate_id],
        edge_input_labels=[],
        cloud_input_labels=[],
        cloud_output_labels=[],
        estimated_edge_flops=0.0,
        estimated_cloud_flops=0.0,
        estimated_payload_bytes=100,
        estimated_privacy_risk=0.0,
        estimated_latency=0.0,
        is_trainable_tail=True,
        legacy_layer_index=prefix,
        boundary_count=1,
        edge_parameter_count=prefix,
        total_parameter_count=total,
        edge_parameter_ratio=ratio,
    )


def test_parse_args_uses_requested_defaults():
    args = experiments._parse_args([])

    assert args.yaml_path == "./config/config.yaml"
    assert args.video_path == "./video_data/road.mp4"
    assert args.edge_model == "rfdetr_nano"
    assert args.golden_model == "rtdetr_x"
    assert args.sample_counts == [64, 128, 256]
    assert args.epochs == [1, 3, 5, 10, 20]
    assert args.batch_size == 32
    assert args.boundary_quantiles == [0.25, 0.5, 0.75]
    assert args.modes == ["full", "freeze", "split_cached", "split_rebuild"]
    assert args.repeats == 1


def test_seeded_frame_selection_is_deterministic_and_nested():
    first = experiments._select_sample_frame_ids(20, [3, 7], seed=11)
    second = experiments._select_sample_frame_ids(20, [3, 7], seed=11)

    assert first == second
    assert set(first[3]).issubset(set(first[7]))
    assert len(first[3]) == 3
    assert len(first[7]) == 7


def test_make_trace_input_repeats_single_sample_for_dynamic_trace():
    sample = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)

    repeated = experiments._make_trace_input(sample, 2)

    assert repeated.shape == (2, 3, 2, 2)
    assert torch.equal(repeated[0], sample[0])
    assert torch.equal(repeated[1], sample[0])


def test_split_boundary_payload_batch_slices_scaled_leading_dims():
    payload = experiments.BoundaryPayload(
        split_id="after:node",
        graph_signature="graph",
        batch_size=2,
        tensors={
            "flat": torch.arange(8).reshape(4, 2),
            "batched": torch.arange(6).reshape(2, 3),
        },
        schema={},
        requires_grad={"flat": False, "batched": False},
        passthrough_inputs={},
    )

    first, second = experiments._split_boundary_payload_batch(payload, batch_size=2)

    assert first.batch_size == second.batch_size == 1
    assert first.tensors["flat"].shape == (2, 2)
    assert second.tensors["flat"].tolist() == [[4, 5], [6, 7]]
    assert first.tensors["batched"].tolist() == [[0, 1, 2]]
    assert second.tensors["batched"].tolist() == [[3, 4, 5]]


def test_candidate_choice_selects_nearest_quantile_and_auto():
    candidates = [
        _candidate("c10", 0.10),
        _candidate("c26", 0.26),
        _candidate("c49", 0.49),
        _candidate("c77", 0.77),
    ]

    choices = experiments._select_candidate_choices(
        candidates,
        auto_candidate=candidates[1],
        boundary_quantiles=[0.25, 0.50, 0.75],
    )

    assert [(choice.bucket, choice.candidate.candidate_id) for choice in choices] == [
        ("Early", "c26"),
        ("Middle", "c49"),
        ("Late", "c77"),
        ("Auto", "c26"),
    ]


def test_result_writers_emit_jsonl_and_summary_csv(tmp_path):
    rows = [
        {
            "mode": "full",
            "success": True,
            "sampled_frame_indices": [1, 5],
            "metrics": {"training_time": 1.25},
        },
        {
            "mode": "freeze",
            "success": False,
            "failure_reason": "boom",
        },
    ]

    jsonl_path = tmp_path / "results.jsonl"
    for row in rows:
        experiments._append_jsonl(jsonl_path, row)
    experiments._write_summary_csv(tmp_path / "summary.csv", rows)

    loaded = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert loaded == rows
    summary_text = (tmp_path / "summary.csv").read_text()
    assert "failure_reason" in summary_text
    assert "boom" in summary_text


def test_aggregate_rows_reports_mean_std_and_success_rate():
    rows = [
        {
            "mode": "freeze",
            "split_bucket": "Auto",
            "candidate_id": "c1",
            "sample_count": 2,
            "epochs": 1,
            "success": True,
            "training_time": 1.0,
            "delta proxy_mAP@0.5": 0.1,
        },
        {
            "mode": "freeze",
            "split_bucket": "Auto",
            "candidate_id": "c1",
            "sample_count": 2,
            "epochs": 1,
            "success": True,
            "training_time": 3.0,
            "delta proxy_mAP@0.5": 0.3,
        },
        {
            "mode": "freeze",
            "split_bucket": "Auto",
            "candidate_id": "c1",
            "sample_count": 2,
            "epochs": 1,
            "success": False,
            "training_time": 999.0,
        },
    ]

    aggregate = experiments._aggregate_rows(rows)

    assert len(aggregate) == 1
    assert aggregate[0]["run_count"] == 3
    assert aggregate[0]["success_count"] == 2
    assert aggregate[0]["failure_count"] == 1
    assert aggregate[0]["success_rate"] == pytest.approx(2 / 3)
    assert aggregate[0]["training_time_mean"] == pytest.approx(2.0)
    assert aggregate[0]["training_time_std"] == pytest.approx(2**0.5)
    assert aggregate[0]["delta proxy_mAP@0.5_mean"] == pytest.approx(0.2)


def test_time_reduction_rows_match_freeze_by_repeat_and_candidate():
    rows = [
        {
            "mode": "freeze",
            "split_bucket": "Auto",
            "candidate_id": "c1",
            "sample_count": 2,
            "epochs": 1,
            "repeat_index": 0,
            "success": True,
            "training_time": 10.0,
        },
        {
            "mode": "freeze",
            "split_bucket": "Auto",
            "candidate_id": "c1",
            "sample_count": 2,
            "epochs": 1,
            "repeat_index": 1,
            "success": True,
            "training_time": 12.0,
        },
        {
            "mode": "split_cached",
            "split_bucket": "Auto",
            "candidate_id": "c1",
            "sample_count": 2,
            "epochs": 1,
            "repeat_index": 0,
            "success": True,
            "training_time": 4.0,
        },
        {
            "mode": "split_cached",
            "split_bucket": "Auto",
            "candidate_id": "c1",
            "sample_count": 2,
            "epochs": 1,
            "repeat_index": 1,
            "success": True,
            "training_time": 5.0,
        },
        {
            "mode": "split_rebuild",
            "split_bucket": "Auto",
            "candidate_id": "c1",
            "sample_count": 2,
            "epochs": 1,
            "repeat_index": 0,
            "success": True,
            "training_time": 6.5,
        },
        {
            "mode": "split_rebuild",
            "split_bucket": "Auto",
            "candidate_id": "c1",
            "sample_count": 2,
            "epochs": 1,
            "repeat_index": 1,
            "success": False,
            "training_time": 99.0,
        },
    ]

    reductions = experiments._time_reduction_rows(rows)

    assert [
        (
            row["mode"],
            row["repeat_index"],
            row["reduction"],
        )
        for row in reductions
    ] == [
        ("split_cached", 0, pytest.approx(6.0)),
        ("split_cached", 1, pytest.approx(7.0)),
        ("split_rebuild", 0, pytest.approx(3.5)),
    ]


def test_training_time_reduction_boxplot_writes_pdf(tmp_path):
    rows = []
    for repeat_index, freeze_time in enumerate([10.0, 12.0, 11.0]):
        rows.extend(
            [
                {
                    "mode": "freeze",
                    "split_bucket": "Auto",
                    "candidate_id": "c1",
                    "sample_count": 2,
                    "epochs": 1,
                    "repeat_index": repeat_index,
                    "success": True,
                    "training_time": freeze_time,
                },
                {
                    "mode": "split_cached",
                    "split_bucket": "Auto",
                    "candidate_id": "c1",
                    "sample_count": 2,
                    "epochs": 1,
                    "repeat_index": repeat_index,
                    "success": True,
                    "training_time": 4.0 + repeat_index,
                },
                {
                    "mode": "split_rebuild",
                    "split_bucket": "Auto",
                    "candidate_id": "c1",
                    "sample_count": 2,
                    "epochs": 1,
                    "repeat_index": repeat_index,
                    "success": True,
                    "training_time": 5.0 + repeat_index,
                },
            ]
        )

    experiments._write_training_time_reduction_boxplot(rows, tmp_path)

    pdf_path = tmp_path / "plots" / "training_time_reduction_boxplot.pdf"
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0


def test_split_position_time_accuracy_plot_writes_pdf_and_png(tmp_path):
    rows = []
    for repeat_index in range(3):
        for mode, training_time, accuracy in [
            ("freeze", 12.0 + repeat_index, 0.82 + 0.01 * repeat_index),
            ("split_cached", 5.0 + repeat_index, 0.78 + 0.01 * repeat_index),
            ("split_rebuild", 6.0 + repeat_index, 0.78 + 0.01 * repeat_index),
        ]:
            rows.append(
                {
                    "mode": mode,
                    "split_bucket": "Early",
                    "candidate_id": "c1",
                    "sample_count": 2,
                    "epochs": 1,
                    "repeat_index": repeat_index,
                    "success": True,
                    "training_time": training_time,
                    "proxy_mAP@0.5 after": accuracy,
                    "prefix_parameter_ratio": 0.25,
                }
            )

    experiments._write_split_position_time_accuracy_plot(rows, tmp_path)

    pdf_path = tmp_path / "plots" / "split_position_time_accuracy_dual_axis.pdf"
    png_path = tmp_path / "plots" / "split_position_time_accuracy_dual_axis.png"
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0
    assert png_path.exists()
    assert png_path.stat().st_size > 0


def test_suffix_parameter_resolution_is_identical_for_freeze_and_split():
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))

    def collect_tail(_runtime):
        params = []
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("1."))
            if parameter.requires_grad:
                params.append(parameter)
        return params

    freeze_params, freeze_names = experiments._resolve_suffix_trainable_parameters(
        model,
        SimpleNamespace(),
        collector=collect_tail,
    )
    split_params, split_names = experiments._resolve_suffix_trainable_parameters(
        model,
        SimpleNamespace(),
        collector=collect_tail,
    )

    assert freeze_names == split_names == ["1.weight", "1.bias"]
    assert sum(parameter.numel() for parameter in freeze_params) == sum(
        parameter.numel() for parameter in split_params
    )


def test_no_prefix_guard_blocks_and_restores_prefix_execution():
    class Runtime:
        def run_prefix(self):
            return "runtime-prefix"

    class Splitter:
        def __init__(self):
            self.runtime = Runtime()

        def _ensure_runtime(self):
            return self.runtime

        def edge_forward(self):
            return "edge-prefix"

        def run_prefix(self):
            return "splitter-prefix"

    splitter = Splitter()

    with experiments._forbid_prefix_execution(splitter):
        with pytest.raises(RuntimeError, match="Prefix forward is forbidden"):
            splitter.runtime.run_prefix()
        with pytest.raises(RuntimeError, match="Prefix forward is forbidden"):
            splitter.edge_forward()
        with pytest.raises(RuntimeError, match="Prefix forward is forbidden"):
            splitter.run_prefix()

    assert splitter.runtime.run_prefix() == "runtime-prefix"
    assert splitter.edge_forward() == "edge-prefix"
    assert splitter.run_prefix() == "splitter-prefix"


def test_mark_failure_records_reason_without_raising():
    row = {"success": True, "failure_reason": None}

    experiments._mark_failure(row, RuntimeError("candidate cannot replay"))

    assert row == {
        "success": False,
        "failure_reason": "candidate cannot replay",
    }
