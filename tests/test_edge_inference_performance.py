from __future__ import annotations

import io
import json
import threading
import time
from queue import Full
from types import SimpleNamespace

import pytest
import torch

from edge.edge_worker import (
    AsyncSampleCollector,
    EdgeWorker,
    SampleCollectionJob,
    _suffix_thread_candidates,
)
from edge.info import TASK_STATE
from edge.task import Task
from edge_client import _write_task_result
from model_management.detectors import legacy_split_model_adapters as split_adapters
from model_management.inference.artifacts import InferenceArtifacts


def _collection_job(sample_id: str = "sample-1") -> SampleCollectionJob:
    inference = InferenceArtifacts(
        intermediate=None,
        final_detection_boxes=[],
        final_detection_labels=[],
        final_detection_scores=[],
        low_threshold_boxes=[],
        low_threshold_labels=[],
        low_threshold_scores=[],
        confidence=0.0,
    )
    return SampleCollectionJob(
        sample_id=sample_id,
        frame_index=1,
        frame=None,
        inference=inference,
        split_config_id="split-a",
        model_id="model-a",
        model_version="0",
        front_version="0",
        split_key="after:test",
        feature_abi_id="abi-a",
        runtime_contract={"feature_layout_id": "abi-a"},
    )


def test_write_task_result_includes_latency_and_timing() -> None:
    task = Task(1, 7, None, 10.0, (1, 1, 3))
    task.end_time = 10.125
    task.state = TASK_STATE.FINISHED
    task.result_source = "inference"
    task.timing_ms = {"split_prefix": 12.5, "total": 125.0}
    task.add_result([[1, 2, 3, 4]], [5], [0.9])

    handle = io.StringIO()
    _write_task_result(handle, task)

    payload = json.loads(handle.getvalue())
    assert payload["frame_index"] == 7
    assert payload["latency_ms"] == pytest.approx(125.0)
    assert payload["timing_ms"] == {"split_prefix": 12.5, "total": 125.0}
    assert payload["result"] == {
        "labels": [5],
        "boxes": [[1, 2, 3, 4]],
        "scores": [0.9],
    }


def test_suffix_thread_candidate_resolution() -> None:
    mode, candidates = _suffix_thread_candidates(
        "auto",
        current_threads=16,
        cpu_count=16,
    )
    assert mode == "auto"
    assert max(candidates) <= 12
    assert 16 not in candidates
    assert 8 in candidates
    assert 12 in candidates
    assert len(candidates) == len(set(candidates))

    true_mode, true_candidates = _suffix_thread_candidates(
        True,
        current_threads=16,
        cpu_count=16,
    )
    assert true_mode == "auto"
    assert true_candidates == candidates

    assert _suffix_thread_candidates(6, current_threads=16, cpu_count=16) == ("fixed", [6])
    assert _suffix_thread_candidates(False, current_threads=16, cpu_count=16) == ("off", [])
    assert _suffix_thread_candidates("off", current_threads=16, cpu_count=16) == ("off", [])
    assert _suffix_thread_candidates("nope", current_threads=16, cpu_count=16) == (
        "invalid",
        [],
    )


def test_split_observables_can_skip_feature_spectral_entropy(monkeypatch) -> None:
    calls = {"feature_entropy": 0}

    def fake_feature_entropy(_payload):
        calls["feature_entropy"] += 1
        return 0.25

    def fake_runtime_logits(_model, _outputs):
        return torch.tensor([[4.0, 1.0, 0.0], [0.5, 2.0, 0.0]]), "softmax"

    monkeypatch.setattr(
        split_adapters,
        "_summarize_payload_spectral_entropy",
        fake_feature_entropy,
    )
    monkeypatch.setattr(split_adapters, "_extract_runtime_logits", fake_runtime_logits)

    skipped = split_adapters.summarize_split_runtime_observables(
        torch.nn.Identity(),
        outputs=None,
        split_payload={"payload": torch.ones(2, 4)},
        include_feature_spectral_entropy=False,
    )

    assert calls["feature_entropy"] == 0
    assert skipped["feature_spectral_entropy"] is None
    assert skipped["logit_entropy"] is not None
    assert skipped["logit_margin"] is not None

    default = split_adapters.summarize_split_runtime_observables(
        torch.nn.Identity(),
        outputs=None,
        split_payload={"payload": torch.ones(2, 4)},
    )

    assert calls["feature_entropy"] == 1
    assert default["feature_spectral_entropy"] == 0.25
    assert default["logit_entropy"] == skipped["logit_entropy"]


def test_async_sample_collector_fifo_flush_and_close() -> None:
    seen: list[str] = []
    entered = threading.Event()
    release = threading.Event()

    def handler(job: SampleCollectionJob) -> None:
        if job.sample_id == "first":
            entered.set()
            assert release.wait(timeout=2.0)
        seen.append(job.sample_id)

    collector = AsyncSampleCollector(handler, maxsize=1)
    try:
        collector.submit_nowait(_collection_job("first"))
        assert entered.wait(timeout=2.0)
        collector.submit_nowait(_collection_job("second"))
        with pytest.raises(Full):
            collector.submit_nowait(_collection_job("third"))

        release.set()
        assert collector.flush(timeout=2.0)
    finally:
        release.set()
        assert collector.close(timeout=2.0)

    assert seen == ["first", "second"]


def test_async_sample_collector_close_timeout_does_not_block_when_queue_full() -> None:
    seen: list[str] = []
    entered = threading.Event()
    release = threading.Event()

    def handler(job: SampleCollectionJob) -> None:
        if job.sample_id == "first":
            entered.set()
            assert release.wait(timeout=2.0)
        seen.append(job.sample_id)

    collector = AsyncSampleCollector(handler, maxsize=1)
    try:
        collector.submit_nowait(_collection_job("first"))
        assert entered.wait(timeout=2.0)
        collector.submit_nowait(_collection_job("second"))

        started = time.perf_counter()
        assert collector.close(timeout=0.05) is False
        assert time.perf_counter() - started < 0.5

        release.set()
        assert collector.close(timeout=2.0)
    finally:
        release.set()
        collector.close(timeout=2.0)

    assert seen == ["first", "second"]


def test_sample_collection_falls_back_synchronously_when_queue_is_full() -> None:
    class FullCollector:
        def submit_nowait(self, _job):
            raise Full

    handled: list[str] = []
    worker = SimpleNamespace(
        sample_collector=FullCollector(),
        _collect_data_from_job=lambda job: handled.append(job.sample_id),
    )

    queued = EdgeWorker._submit_sample_collection(worker, _collection_job("fallback"))

    assert queued is False
    assert handled == ["fallback"]
