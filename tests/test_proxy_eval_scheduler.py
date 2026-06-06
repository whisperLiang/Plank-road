from __future__ import annotations

from cloud.training.proxy_eval import (
    ProxyEarlyStopper,
    ProxyEvalConfig,
    ProxyEvalScheduler,
    deterministic_proxy_sample_ids,
)
from cloud.training.types import ProxyEvalResult


def test_proxy_scheduler_evaluates_epoch_one_by_default() -> None:
    scheduler = ProxyEvalScheduler(ProxyEvalConfig(interval_epochs=10))

    assert scheduler.should_eval(1, 50)


def test_proxy_scheduler_evaluates_interval_epochs_only_when_not_final() -> None:
    scheduler = ProxyEvalScheduler(
        ProxyEvalConfig(eval_after_first_epoch=False, eval_final=False, interval_epochs=10)
    )

    evaluated = [epoch for epoch in range(1, 36) if scheduler.should_eval(epoch, 35)]

    assert evaluated == [10, 20, 30]


def test_proxy_scheduler_final_epoch_is_configurable() -> None:
    enabled = ProxyEvalScheduler(
        ProxyEvalConfig(eval_after_first_epoch=False, eval_final=True, interval_epochs=10)
    )
    disabled = ProxyEvalScheduler(
        ProxyEvalConfig(eval_after_first_epoch=False, eval_final=False, interval_epochs=10)
    )

    assert enabled.should_eval(35, 35)
    assert not disabled.should_eval(35, 35)


def test_deterministic_proxy_sample_ids_are_stable_and_bounded() -> None:
    annotations = {"sample-10": {}, "sample-2": {}, "a": {}, "z": {}}

    assert deterministic_proxy_sample_ids(annotations, 2) == ["a", "sample-10"]
    assert deterministic_proxy_sample_ids(annotations, 0) == [
        "a",
        "sample-10",
        "sample-2",
        "z",
    ]
    assert deterministic_proxy_sample_ids(dict(reversed(list(annotations.items()))), 2) == [
        "a",
        "sample-10",
    ]


def test_proxy_sample_ids_prioritize_relabels_then_stable_random_fill() -> None:
    annotations = {f"sample-{index:02d}": {} for index in range(8)}
    priority = ["sample-06", "sample-01"]

    selected = deterministic_proxy_sample_ids(
        annotations,
        5,
        priority_sample_ids=priority,
        random_fill_seed="view-a",
    )
    selected_again = deterministic_proxy_sample_ids(
        dict(reversed(list(annotations.items()))),
        5,
        priority_sample_ids=list(reversed(priority)),
        random_fill_seed="view-a",
    )

    assert selected == selected_again
    assert len(selected) == 5
    assert set(selected[:2]) == set(priority)
    assert set(selected[2:]).isdisjoint(priority)
    assert len(set(selected)) == len(selected)


def test_proxy_early_stopper_uses_patience_and_min_delta() -> None:
    config = ProxyEvalConfig(min_delta=0.002, patience=2)
    stopper = ProxyEarlyStopper(config, baseline_metric=0.5)

    first = stopper.record(
        ProxyEvalResult({"map": 0.501}, 0.501, 0.0, epoch=1),
        improved=False,
        best_metric=0.5,
    )
    second = stopper.record(
        ProxyEvalResult({"map": 0.5015}, 0.5015, 0.0, epoch=10),
        improved=False,
        best_metric=0.5,
    )

    assert not first.should_stop
    assert second.should_stop


def test_proxy_early_stopper_high_baseline_stops_faster() -> None:
    config = ProxyEvalConfig(min_delta=0.002, patience=2, skip_if_baseline_above=0.98)
    stopper = ProxyEarlyStopper(config, baseline_metric=0.99)

    decision = stopper.record(
        ProxyEvalResult({"map": 0.9905}, 0.9905, 0.0, epoch=1),
        improved=False,
        best_metric=0.99,
    )

    assert decision.should_stop
