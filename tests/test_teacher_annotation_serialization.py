from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from loguru import logger

from cloud_server import CloudContinualLearner
from model_management.object_detection import Object_Detection


def _build_learner(tmp_path, *, teacher_batch_size: int = 2) -> CloudContinualLearner:
    config = SimpleNamespace(
        edge_model_name="rfdetr_nano",
        continual_learning=SimpleNamespace(
            batch_size=teacher_batch_size,
            teacher_batch_size=teacher_batch_size,
        ),
        das=SimpleNamespace(enabled=False),
        workspace_root=str(tmp_path),
    )
    return CloudContinualLearner(
        config=config,
        large_object_detection=SimpleNamespace(),
    )


def _write_frames(frame_dir: Path, sample_ids) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for sample_id in sample_ids:
        frame = np.full((12, 12, 3), 80, dtype=np.uint8)
        assert cv2.imwrite(str(frame_dir / f"{sample_id}.jpg"), frame)


def _prediction(label: int = 1):
    return ([[1.0, 1.0, 8.0, 8.0]], [label], [0.95])


@contextmanager
def _captured_log_messages():
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]), level="INFO")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


def _assert_teacher_slot_logs_if_present(messages: list[str]) -> None:
    joined = " ".join(message.lower() for message in messages)
    if "teacher" not in joined or "slot" not in joined:
        return
    assert "waiting" in joined
    assert "acquired" in joined
    assert "released" in joined


def _join_without_errors(thread: threading.Thread, errors: list[BaseException]) -> None:
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert not errors


def test_collect_teacher_annotations_serializes_globally_across_requests(tmp_path):
    learner_first = _build_learner(tmp_path / "learner-a", teacher_batch_size=2)
    learner_second = _build_learner(tmp_path / "learner-b", teacher_batch_size=2)

    frame_dir_first = tmp_path / "frames-a"
    frame_dir_second = tmp_path / "frames-b"
    _write_frames(frame_dir_first, ["0", "1"])
    _write_frames(frame_dir_second, ["10", "11"])

    lock = threading.Lock()
    first_entered = threading.Event()
    second_entered = threading.Event()
    allow_first_finish = threading.Event()
    overlap_detected = threading.Event()
    order: list[tuple[str, str, int]] = []
    active_calls = {"count": 0}
    errors: list[BaseException] = []

    def _make_stub(name: str):
        def _stub(frames):
            with lock:
                active_calls["count"] += 1
                order.append((name, "enter", len(frames)))
                if active_calls["count"] > 1:
                    overlap_detected.set()
            if name == "first":
                first_entered.set()
                assert allow_first_finish.wait(timeout=2.0)
            else:
                second_entered.set()
            time.sleep(0.01)
            with lock:
                active_calls["count"] -= 1
                order.append((name, "exit", len(frames)))
            return [_prediction(index + 1) for index, _ in enumerate(frames)]

        return _stub

    learner_first._teacher_inference_batch = _make_stub("first")
    learner_second._teacher_inference_batch = _make_stub("second")

    def _run_first():
        try:
            learner_first._collect_teacher_annotations(str(frame_dir_first), ["0", "1"])
        except BaseException as exc:  # pragma: no cover - assertion path
            errors.append(exc)

    def _run_second():
        try:
            learner_second._collect_teacher_annotations(str(frame_dir_second), ["10", "11"])
        except BaseException as exc:  # pragma: no cover - assertion path
            errors.append(exc)

    with _captured_log_messages() as messages:
        thread_first = threading.Thread(target=_run_first, name="teacher-first")
        thread_second = threading.Thread(target=_run_second, name="teacher-second")
        thread_first.start()
        assert first_entered.wait(timeout=2.0)

        thread_second.start()
        time.sleep(0.1)

        assert thread_second.is_alive()
        assert not second_entered.is_set()
        assert not overlap_detected.is_set()

        allow_first_finish.set()
        _join_without_errors(thread_first, errors)
        _join_without_errors(thread_second, errors)

    assert order == [
        ("first", "enter", 2),
        ("first", "exit", 2),
        ("second", "enter", 2),
        ("second", "exit", 2),
    ]
    _assert_teacher_slot_logs_if_present(messages)


def test_generate_annotations_uses_same_global_teacher_serialization(tmp_path):
    learner_first = _build_learner(tmp_path / "legacy-a", teacher_batch_size=2)
    learner_second = _build_learner(tmp_path / "legacy-b", teacher_batch_size=2)

    cache_a = tmp_path / "cache-a"
    cache_b = tmp_path / "cache-b"
    _write_frames(cache_a / "frames", [1, 2])
    _write_frames(cache_b / "frames", [3, 4])

    lock = threading.Lock()
    first_entered = threading.Event()
    second_entered = threading.Event()
    allow_first_finish = threading.Event()
    overlap_detected = threading.Event()
    order: list[tuple[str, str]] = []
    active_calls = {"count": 0}
    errors: list[BaseException] = []

    def _make_stub(name: str):
        def _stub(_frame):
            with lock:
                active_calls["count"] += 1
                order.append((name, "enter"))
                if active_calls["count"] > 1:
                    overlap_detected.set()
            if name == "first":
                first_entered.set()
                assert allow_first_finish.wait(timeout=2.0)
            else:
                second_entered.set()
            time.sleep(0.01)
            with lock:
                active_calls["count"] -= 1
                order.append((name, "exit"))
            return _prediction(1)

        return _stub

    learner_first._teacher_inference = _make_stub("first")
    learner_second._teacher_inference = _make_stub("second")

    def _run_first():
        try:
            learner_first._generate_annotations(edge_id=1, frame_indices=[1, 2], cache_path=str(cache_a))
        except BaseException as exc:  # pragma: no cover - assertion path
            errors.append(exc)

    def _run_second():
        try:
            learner_second._generate_annotations(edge_id=2, frame_indices=[3, 4], cache_path=str(cache_b))
        except BaseException as exc:  # pragma: no cover - assertion path
            errors.append(exc)

    with _captured_log_messages() as messages:
        thread_first = threading.Thread(target=_run_first, name="legacy-first")
        thread_second = threading.Thread(target=_run_second, name="legacy-second")
        thread_first.start()
        assert first_entered.wait(timeout=2.0)

        thread_second.start()
        time.sleep(0.1)

        assert thread_second.is_alive()
        assert not second_entered.is_set()
        assert not overlap_detected.is_set()

        allow_first_finish.set()
        _join_without_errors(thread_first, errors)
        _join_without_errors(thread_second, errors)

    first_second_index = next(index for index, item in enumerate(order) if item[0] == "second")
    assert all(item[0] == "first" for item in order[:first_second_index])
    assert all(item[0] == "second" for item in order[first_second_index:])
    _assert_teacher_slot_logs_if_present(messages)


def test_collect_teacher_annotations_keeps_request_level_batching(tmp_path, monkeypatch):
    learner = _build_learner(tmp_path / "batched", teacher_batch_size=2)
    frame_dir = tmp_path / "batched-frames"
    _write_frames(frame_dir, ["0", "1", "2"])

    batch_sizes: list[int] = []

    def _batch_stub(frames):
        batch_sizes.append(len(frames))
        return [_prediction(index + 1) for index, _ in enumerate(frames)]

    monkeypatch.setattr(learner, "_teacher_inference_batch", _batch_stub)
    monkeypatch.setattr(
        learner,
        "_teacher_inference",
        lambda _frame: pytest.fail("Expected batched teacher inference, not per-image fallback."),
    )

    annotations = learner._collect_teacher_annotations(str(frame_dir), ["0", "1", "2"])

    assert batch_sizes == [2, 1]
    assert sorted(annotations) == ["0", "1", "2"]


def test_collect_teacher_annotations_rejects_invalid_batch_without_single_fallback(
    tmp_path,
    monkeypatch,
):
    learner = _build_learner(tmp_path / "invalid-batch", teacher_batch_size=2)
    frame_dir = tmp_path / "invalid-batch-frames"
    _write_frames(frame_dir, ["0", "1"])

    monkeypatch.setattr(learner, "_teacher_inference_batch", lambda frames: [_prediction(1)])
    monkeypatch.setattr(
        learner,
        "_teacher_inference",
        lambda _frame: pytest.fail("Expected batch-level failure, not per-image fallback."),
    )

    with pytest.raises(RuntimeError, match="invalid result count"):
        learner._collect_teacher_annotations(str(frame_dir), ["0", "1"])


def test_collect_teacher_annotations_requires_batch_teacher_without_single_fallback(
    tmp_path,
    monkeypatch,
):
    learner = _build_learner(tmp_path / "missing-batch", teacher_batch_size=2)
    frame_dir = tmp_path / "missing-batch-frames"
    _write_frames(frame_dir, ["0", "1"])
    learner.large_od = SimpleNamespace(
        large_inference=lambda _frame, **_kwargs: pytest.fail(
            "Expected missing batch inference to fail, not run single-sample inference."
        )
    )

    with pytest.raises(RuntimeError, match="large_inference_batch"):
        learner._collect_teacher_annotations(str(frame_dir), ["0", "1"])


def test_hot_teacher_inference_paths_do_not_clear_cuda_cache(monkeypatch):
    detector = Object_Detection.__new__(Object_Detection)
    detector.threshold_high = 0.6
    detector._prepare_image_tensor = lambda frame: torch.as_tensor(frame, dtype=torch.float32)
    detector._parse_prediction_output = lambda res, threshold: (res, threshold, "parsed")
    detector.model = lambda images: [
        {
            "labels": torch.tensor([1], dtype=torch.int64),
            "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
            "scores": torch.tensor([0.9], dtype=torch.float32),
        }
        for _ in images
    ]

    empty_cache_calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_cache_calls.append("called"))

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    detector.large_inference_batch([frame], threshold=0.5)
    detector.get_model_prediction(frame, threshold=0.5)

    assert empty_cache_calls == []
