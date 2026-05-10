from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

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
