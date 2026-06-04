from __future__ import annotations

import json
import time
from dataclasses import replace

import cv2
import numpy as np

from cloud.annotation import (
    TeacherAnnotationRequest,
    TeacherAnnotationService,
    TeacherAnnotationWorker,
    TeacherLabelCache,
)


def _image(tmp_path, name: str) -> str:
    path = tmp_path / f"{name}.jpg"
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), frame)
    return str(path)


def _request(tmp_path, sample_id: str = "sample-1", **overrides) -> TeacherAnnotationRequest:
    payload = {
        "sample_id": sample_id,
        "edge_id": 1,
        "model_id": "yolo26n",
        "image_path": _image(tmp_path, sample_id),
        "image_sha1": "b" * 40,
        "teacher_model_name": "rtdetr_x",
        "teacher_weights_fingerprint": "weights-a",
        "teacher_label_schema": "coco_91",
        "teacher_num_classes": 91,
        "teacher_annotation_threshold": 0.5,
        "label_coordinate_space": "original_xyxy",
        "label_runtime_version": "fixed-split-pool-labels.v1",
        "metadata": {"include_empty": True},
    }
    payload.update(overrides)
    return TeacherAnnotationRequest(**payload)


def _prediction(_frame):
    return ([[1, 2, 3, 4]], [7], [0.9])


def test_cache_hit_does_not_submit_to_worker(tmp_path) -> None:
    cache = TeacherLabelCache(str(tmp_path / "cache"))
    req = _request(tmp_path)
    cache.write(req, {"boxes": [], "labels": []}, source="test")
    worker = TeacherAnnotationWorker(label_cache=cache, auto_start=False)
    service = TeacherAnnotationService(label_cache=cache, worker=worker)

    result = service.ensure_many([req], wait=True, timeout_sec=0.01)

    assert result.cache_hits == 1
    assert result.submitted == 0
    assert result.labels_by_sample_id["sample-1"] == {"boxes": [], "labels": []}


def test_cache_miss_submits_worker_and_stays_unresolved_until_worker_finishes(tmp_path) -> None:
    cache = TeacherLabelCache(str(tmp_path / "cache"))
    worker = TeacherAnnotationWorker(label_cache=cache, auto_start=False)
    service = TeacherAnnotationService(label_cache=cache, worker=worker)
    req = _request(tmp_path)

    result = service.ensure_many([req], wait=False, timeout_sec=None)

    assert result.submitted == 1
    assert result.unresolved_count == 1
    assert cache.read(req) is None


def test_worker_result_is_returned_from_cache(tmp_path) -> None:
    cache = TeacherLabelCache(str(tmp_path / "cache"))
    worker = TeacherAnnotationWorker(
        label_cache=cache,
        batch_inference=lambda images, threshold: [_prediction(image) for image in images],
        worker_batch_size=4,
        auto_start=False,
    )
    service = TeacherAnnotationService(label_cache=cache, worker=worker)
    req = _request(tmp_path)

    service.submit_many([req])
    worker.process_pending_once()
    result = service.ensure_many([req], wait=False, timeout_sec=None)

    assert result.cache_hits == 1
    assert result.labels_by_sample_id["sample-1"] == {
        "boxes": [[1, 2, 3, 4]],
        "labels": [7],
        "scores": [0.9],
    }


def test_invalid_cache_metadata_is_resubmitted_to_worker(tmp_path) -> None:
    cache = TeacherLabelCache(str(tmp_path / "cache"))
    req = _request(tmp_path)
    cache.write(req, {"boxes": [], "labels": []}, source="test")
    meta_path = cache.metadata_path(req.cache_key())
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "cache_version": cache.cache_version,
                "cache_key": "stale",
                "key_payload": {},
            },
            handle,
        )

    worker = TeacherAnnotationWorker(
        label_cache=cache,
        batch_inference=lambda images, threshold: [_prediction(image) for image in images],
        auto_start=False,
    )
    service = TeacherAnnotationService(label_cache=cache, worker=worker)

    result = service.ensure_many([req], wait=False, timeout_sec=None)
    worker.process_pending_once()

    assert result.submitted == 1
    assert cache.read(req) == {
        "boxes": [[1, 2, 3, 4]],
        "labels": [7],
        "scores": [0.9],
    }


def test_same_cache_key_submit_is_deduped(tmp_path) -> None:
    cache = TeacherLabelCache(str(tmp_path / "cache"))
    worker = TeacherAnnotationWorker(label_cache=cache, auto_start=False)
    service = TeacherAnnotationService(label_cache=cache, worker=worker)
    req_a = _request(tmp_path, "a")
    req_b = replace(req_a, sample_id="b")

    result = service.submit_many([req_a, req_b])

    assert result.submitted == 1
    assert result.duplicate == 1


def test_missing_worker_reports_unresolved(tmp_path) -> None:
    cache = TeacherLabelCache(str(tmp_path / "cache"))
    req = _request(tmp_path)
    service = TeacherAnnotationService(label_cache=cache)

    result = service.ensure_many([req], wait=False, timeout_sec=None)

    assert result.unresolved_count == 1
    assert result.unresolved_sample_ids == ["sample-1"]


def test_wait_timeout_returns_after_requested_timeout(tmp_path) -> None:
    cache = TeacherLabelCache(str(tmp_path / "cache"))
    worker = TeacherAnnotationWorker(label_cache=cache, auto_start=False)
    service = TeacherAnnotationService(label_cache=cache, worker=worker)
    req = _request(tmp_path)

    started = time.perf_counter()
    result = service.ensure_many([req], wait=True, timeout_sec=0.02)

    assert time.perf_counter() - started >= 0.015
    assert result.waited_sec >= 0.015
    assert result.unresolved_count == 1


def test_cache_hit_and_worker_result_label_format_match(tmp_path) -> None:
    cache = TeacherLabelCache(str(tmp_path / "cache"))
    labels = {"boxes": [[1, 2, 3, 4]], "labels": [2], "scores": [0.8]}
    cached = _request(tmp_path, "cached", image_sha1="c" * 40)
    worker_req = _request(tmp_path, "worker", image_sha1="d" * 40)
    cache.write(cached, labels, source="test")
    worker = TeacherAnnotationWorker(
        label_cache=cache,
        batch_inference=lambda images, threshold: [
            ([[1, 2, 3, 4]], [2], [0.8]) for _image in images
        ],
        auto_start=False,
    )
    service = TeacherAnnotationService(label_cache=cache, worker=worker)

    service.submit_many([worker_req])
    worker.process_pending_once()
    result = service.ensure_many([cached, worker_req], wait=False, timeout_sec=None)

    assert result.labels_by_sample_id["cached"] == labels
    assert result.labels_by_sample_id["worker"] == labels
