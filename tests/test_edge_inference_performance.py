from __future__ import annotations

import io
import json
import threading
import time
from queue import Full, Queue
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from edge.edge_worker import (
    AsyncSampleCollector,
    AsyncSampleWriter,
    EdgeWorker,
    PendingModelUpdate,
    SampleCollectionJob,
    SampleStatsDelta,
    SampleWriteJob,
    _suffix_thread_candidates,
)
from edge.info import TASK_STATE
from edge.task import Task
from edge_client import _write_buffered_task_result, _write_task_result
from model_management import object_detection as object_detection_module
from model_management.detectors import legacy_split_model_adapters as split_adapters
from model_management.inference.artifacts import InferenceArtifacts
from model_management.object_detection import Object_Detection


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
    task.timing_ms = {"split_prefix_ms": 12.5, "task_complete_ms": 125.0}
    task.add_result([[1, 2, 3, 4]], [5], [0.9])

    handle = io.StringIO()
    _write_task_result(handle, task)

    payload = json.loads(handle.getvalue())
    assert payload["frame_index"] == 7
    assert payload["latency_ms"] == pytest.approx(125.0)
    assert payload["timing_ms"] == {
        "split_prefix_ms": 12.5,
        "task_complete_ms": 125.0,
    }
    assert payload["result"] == {
        "labels": [5],
        "boxes": [[1, 2, 3, 4]],
        "scores": [0.9],
    }


def test_write_task_result_does_not_flush_per_frame() -> None:
    class FlushTrackingBuffer(io.StringIO):
        flush_count = 0

        def flush(self) -> None:
            self.flush_count += 1
            super().flush()

    task = Task(1, 1, None, 1.0, (1, 1, 3))
    task.end_time = 1.01
    task.state = TASK_STATE.FINISHED
    handle = FlushTrackingBuffer()

    _write_task_result(handle, task)

    assert handle.flush_count == 0


def test_buffered_task_result_flushes_every_configured_frame() -> None:
    class FlushTrackingBuffer(io.StringIO):
        flush_count = 0

        def flush(self) -> None:
            self.flush_count += 1
            super().flush()

    handle = FlushTrackingBuffer()
    pending = 0
    for frame_index in range(1, 31):
        task = Task(1, frame_index, None, 1.0, (1, 1, 3))
        task.end_time = 1.01
        task.state = TASK_STATE.FINISHED
        pending = _write_buffered_task_result(
            handle,
            task,
            unflushed_count=pending,
            flush_every_n_frames=30,
        )

    assert handle.flush_count == 1
    assert pending == 0


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


def test_sample_collection_drops_without_sync_fallback_when_queue_is_full() -> None:
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
    assert handled == []


def test_strict_sample_collection_uses_explicit_blocking_submit() -> None:
    submitted: list[str] = []

    class StrictCollector:
        def submit_blocking(self, job):
            submitted.append(job.sample_id)

    worker = SimpleNamespace(
        sample_collector=StrictCollector(),
        strict_sample_collection=True,
    )

    assert EdgeWorker._submit_sample_collection(worker, _collection_job("strict"))
    assert submitted == ["strict"]


def _write_job(sample_id: str, quality_bucket: str) -> SampleWriteJob:
    return SampleWriteJob(
        store_kwargs={
            "sample_id": sample_id,
            "frame_index": 1,
            "quality_bucket": quality_bucket,
        },
        stats_delta=SampleStatsDelta.from_values(quality_bucket=quality_bucket),
    )


def test_async_writer_replaces_queued_low_quality_for_high_quality() -> None:
    entered = threading.Event()
    release = threading.Event()
    stored: list[str] = []

    class BlockingStore:
        def store_sample(self, **kwargs):
            if kwargs["sample_id"] == "active-low":
                entered.set()
                assert release.wait(timeout=2.0)
            stored.append(kwargs["sample_id"])
            return SimpleNamespace(**kwargs)

    writer = AsyncSampleWriter(BlockingStore(), maxsize=1)
    try:
        accepted, dropped = writer.submit_nowait(_write_job("active-low", "low_quality"))
        assert accepted and dropped is None
        assert entered.wait(timeout=2.0)
        accepted, dropped = writer.submit_nowait(_write_job("queued-low", "low_quality"))
        assert accepted and dropped is None

        started = time.perf_counter()
        accepted, dropped = writer.submit_nowait(_write_job("high", "high_quality"))

        assert time.perf_counter() - started < 0.1
        assert accepted
        assert dropped is not None
        assert dropped.store_kwargs["sample_id"] == "queued-low"
        release.set()
        assert writer.flush(timeout=2.0)
    finally:
        release.set()
        writer.close(timeout=2.0)

    assert stored == ["active-low", "high"]


def test_split_inference_summarizes_observables_once_without_second_replay(monkeypatch) -> None:
    detector = Object_Detection.__new__(Object_Detection)
    detector.model = torch.nn.Identity()
    detector.model_lock = threading.Lock()
    detector._split_input_resize_mode = "direct_resize"
    detector.threshold_low = 0.1
    detector.config = SimpleNamespace(final_detection_threshold=0.5)
    detector._parse_prediction_output = lambda _output, _threshold: ([], [], [])

    monkeypatch.setattr(
        object_detection_module,
        "prepare_split_runtime_input",
        lambda *_args, **_kwargs: torch.ones(1, 3, 4, 4),
    )
    monkeypatch.setattr(
        object_detection_module,
        "postprocess_split_runtime_output",
        lambda *_args, **_kwargs: [],
    )
    observables_calls = 0

    def summarize(*_args, **_kwargs):
        nonlocal observables_calls
        observables_calls += 1
        return {
            "feature_spectral_entropy": None,
            "logit_entropy": 0.25,
            "logit_margin": 0.5,
            "logit_energy": 1.0,
        }

    monkeypatch.setattr(
        object_detection_module,
        "summarize_split_runtime_observables",
        summarize,
    )

    class Splitter:
        calls = 0
        outputs = {"pred_logits": torch.ones(1, 1, 2)}

        def replay_inference(self, _input, *, return_split_output, profile):
            self.calls += 1
            assert return_split_output
            profile.update({"split_prefix": 1.0, "split_suffix": 2.0})
            return self.outputs, {"payload": torch.ones(1, 2)}

    splitter = Splitter()
    artifacts = detector.infer_sample(np.zeros((4, 4, 3), dtype=np.uint8), splitter=splitter)

    assert splitter.calls == 1
    assert observables_calls == 1
    assert artifacts.logit_entropy == pytest.approx(0.25)
    assert artifacts.timing_ms["split_prefix_ms"] == pytest.approx(1.0)
    assert artifacts.timing_ms["split_suffix_ms"] == pytest.approx(2.0)


def test_local_worker_marks_task_done_before_sample_collection() -> None:
    worker = EdgeWorker.__new__(EdgeWorker)
    worker._stop_event = threading.Event()
    worker.local_queue = Queue()
    worker.config = SimpleNamespace(wait_thresh=100)
    worker.model_version = "0"
    worker.collect_flag = True
    worker.split_learning_enabled = True
    worker.fixed_split_plan = SimpleNamespace(split_config_id="split")
    worker.latest_result_lock = threading.Lock()
    worker.latest_result = {}
    worker._resolve_active_splitter = lambda *_args: object()
    worker._try_apply_pending_model_update = lambda: False
    inference = InferenceArtifacts(
        intermediate={"payload": torch.ones(1)},
        final_detection_boxes=[[1, 2, 3, 4]],
        final_detection_labels=[1],
        final_detection_scores=[0.9],
        low_threshold_boxes=[[1, 2, 3, 4]],
        low_threshold_labels=[1],
        low_threshold_scores=[0.9],
        confidence=0.9,
        timing_ms={
            "split_preprocess_ms": 1.0,
            "split_prefix_ms": 2.0,
            "split_suffix_ms": 3.0,
            "postprocess_ms": 4.0,
            "parse_filter_ms": 5.0,
        },
    )
    worker.small_object_detection = SimpleNamespace(
        infer_sample=lambda *_args, **_kwargs: inference,
    )
    collection_seen: list[bool] = []

    def collect_data(task, _frame, _inference):
        collection_seen.append(task.done_event.is_set())
        worker._stop_event.set()
        return True

    worker.collect_data = collect_data
    task = Task(
        1,
        1,
        np.zeros((4, 4, 3), dtype=np.uint8),
        time.time(),
        (4, 4, 3),
    )
    task.local_queue_enqueued_perf = time.perf_counter()
    worker.local_queue.put(task)

    worker.local_worker()

    assert task.done_event.is_set()
    assert task.state == TASK_STATE.FINISHED
    assert collection_seen == [True]


def test_task_completion_timestamp_includes_latest_result_snapshot() -> None:
    worker = EdgeWorker.__new__(EdgeWorker)
    worker.model_version = "0"
    snapshot_finished = threading.Event()

    def remember_latest_result(_task):
        time.sleep(0.02)
        snapshot_finished.set()

    worker._remember_latest_result = remember_latest_result
    task = Task(1, 1, None, time.time(), (1, 1, 3))
    task.result_source = "inference"
    task.set_inference_artifacts({"result_source": "inference"})

    EdgeWorker._set_task_terminal_state(
        worker,
        task,
        TASK_STATE.FINISHED,
        result_source="inference",
    )

    assert snapshot_finished.is_set()
    assert task.done_event.is_set()
    assert task.end_time is not None
    assert task.timing_ms["task_complete_ms"] >= 20.0


def test_latest_result_reuses_owned_frame_buffer() -> None:
    worker = EdgeWorker.__new__(EdgeWorker)
    worker.model_version = "0"
    worker.latest_result_lock = threading.Lock()
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    task = Task(1, 1, frame, time.time(), frame.shape)
    task.result_source = "inference"
    task.set_inference_artifacts({"result_source": "inference"})

    EdgeWorker._remember_latest_result(worker, task)

    assert worker.latest_result["frame"] is frame


def test_cached_frame_boundary_attempts_pending_model_update() -> None:
    worker = EdgeWorker.__new__(EdgeWorker)
    worker._stop_event = threading.Event()
    worker.frame_cache = Queue()
    worker.config = SimpleNamespace(diff_flag=True, diff_thresh=1.0)
    worker.diff = 0.0
    worker.edge_processor = SimpleNamespace(
        get_frame_feature=lambda _frame: 0.0,
        cal_frame_diff=lambda _current, _previous: 0.0,
    )
    update_attempted = threading.Event()

    def try_apply():
        update_attempted.set()
        worker._stop_event.set()
        return True

    worker._try_apply_pending_model_update = try_apply
    worker.decision_worker = lambda _task: None
    worker._reuse_latest_result = lambda task: setattr(task, "result_source", "cached")
    worker._finalize_task = lambda task: task.mark_done()
    first = Task(1, 1, np.zeros((2, 2, 3), dtype=np.uint8), time.time(), (2, 2, 3))
    cached = Task(1, 2, np.zeros((2, 2, 3), dtype=np.uint8), time.time(), (2, 2, 3))
    worker.frame_cache.put(first)
    worker.frame_cache.put(cached)

    worker.diff_worker()

    assert cached.done_event.is_set()
    assert update_attempted.is_set()


def test_pending_model_update_defers_when_model_lock_is_busy() -> None:
    worker = EdgeWorker.__new__(EdgeWorker)
    worker._pending_model_update_lock = threading.Lock()
    update = PendingModelUpdate(
        update_payload={},
        state_dict={},
        submitted_model_version="0",
        next_model_version="1",
    )
    worker.pending_model_update = update
    model_lock = threading.Lock()
    worker.small_object_detection = SimpleNamespace(model_lock=model_lock)
    worker._apply_prepared_model_update_locked = lambda item: setattr(
        item, "applied_version", "1"
    )

    model_lock.acquire()
    try:
        assert EdgeWorker._try_apply_pending_model_update(worker) is False
        assert worker.pending_model_update is update
        assert not update.applied_event.is_set()
    finally:
        model_lock.release()

    assert EdgeWorker._try_apply_pending_model_update(worker) is True
    assert worker.pending_model_update is None
    assert update.applied_event.is_set()
