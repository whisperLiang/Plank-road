from __future__ import annotations

import time
from collections.abc import Sequence

from loguru import logger

from cloud.annotation.label_cache import TeacherLabelCache
from cloud.annotation.teacher_worker import TeacherAnnotationWorker
from cloud.annotation.types import (
    TeacherAnnotationEnsureResult,
    TeacherAnnotationRequest,
    TeacherAnnotationResult,
    TeacherAnnotationStatus,
    TeacherAnnotationSubmitResult,
)


class TeacherAnnotationService:
    def __init__(
        self,
        *,
        label_cache: TeacherLabelCache,
        worker: TeacherAnnotationWorker | None = None,
    ) -> None:
        self.label_cache = label_cache
        self.worker = worker

    def lookup_many(
        self,
        requests: Sequence[TeacherAnnotationRequest],
    ) -> TeacherAnnotationSubmitResult:
        requested = list(requests or [])
        results, cache_read_time = self.label_cache.lookup_many(requested)
        cache_hits = sum(1 for result in results if result.status == TeacherAnnotationStatus.CACHE_HIT)
        cache_misses = len(results) - cache_hits
        logger.info(
            "[TeacherAnnotation][CacheHit] requested_samples={} cache_hits={} cache_misses={} cache_read_time={:.3f}s",
            len(requested),
            cache_hits,
            cache_misses,
            cache_read_time,
        )
        if cache_misses:
            logger.info(
                "[TeacherAnnotation][CacheMiss] requested_samples={} cache_misses={}",
                len(requested),
                cache_misses,
            )
        return TeacherAnnotationSubmitResult(
            requested_samples=len(requested),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            results=results,
            cache_read_time=cache_read_time,
        )

    def submit_many(
        self,
        requests: Sequence[TeacherAnnotationRequest],
    ) -> TeacherAnnotationSubmitResult:
        requested = list(requests or [])
        lookup = self.lookup_many(requested)
        misses = [
            result.request
            for result in lookup.results
            if result.status == TeacherAnnotationStatus.CACHE_MISS
        ]
        submitted = 0
        duplicate = 0
        failed_count = 0
        worker_results: list[TeacherAnnotationResult] = []
        if misses and self.worker is not None:
            worker_submit = self.worker.submit_many(misses)
            submitted = int(worker_submit.submitted)
            duplicate = int(worker_submit.duplicate)
            failed_count = int(worker_submit.failed_count)
            worker_results = list(worker_submit.results)
        elif misses:
            worker_results = [
                TeacherAnnotationResult(
                    request=request,
                    status=TeacherAnnotationStatus.UNRESOLVED,
                    error="teacher annotation worker is not configured",
                )
                for request in misses
            ]
            failed_count = len(worker_results)

        results = [
            result
            for result in lookup.results
            if result.status == TeacherAnnotationStatus.CACHE_HIT
        ] + worker_results
        logger.info(
            "[TeacherAnnotation][Submit] requested_samples={} cache_hits={} cache_misses={} "
            "submitted={} duplicate={} failed_count={}",
            len(requested),
            lookup.cache_hits,
            lookup.cache_misses,
            submitted,
            duplicate,
            failed_count,
        )
        return TeacherAnnotationSubmitResult(
            requested_samples=len(requested),
            cache_hits=lookup.cache_hits,
            cache_misses=lookup.cache_misses,
            submitted=submitted,
            duplicate=duplicate,
            failed_count=failed_count,
            results=results,
            cache_read_time=lookup.cache_read_time,
        )

    def ensure_many(
        self,
        requests: Sequence[TeacherAnnotationRequest],
        *,
        wait: bool,
        timeout_sec: float | None,
    ) -> TeacherAnnotationEnsureResult:
        requested = list(requests or [])
        ensure_started = time.perf_counter()
        worker_stats_before = self.worker.snapshot_stats() if self.worker is not None else {}

        initial_lookup = self.lookup_many(requested)
        cache_read_time = initial_lookup.cache_read_time
        results: list[TeacherAnnotationResult] = [
            result
            for result in initial_lookup.results
            if result.status == TeacherAnnotationStatus.CACHE_HIT
        ]
        resolved_ids = {result.sample_id for result in results if result.labels is not None}
        misses = [
            result.request
            for result in initial_lookup.results
            if result.status == TeacherAnnotationStatus.CACHE_MISS
        ]

        submitted = 0
        if misses:
            submit_result = self.submit_many(misses)
            submitted = int(submit_result.submitted)

        waited_sec = 0.0
        if wait and misses and self.worker is not None:
            waited_sec = self.worker.wait_for(misses, timeout_sec=timeout_sec)

        remaining = [request for request in misses if str(request.sample_id) not in resolved_ids]
        if remaining:
            post_lookup = self.lookup_many(remaining)
            cache_read_time += post_lookup.cache_read_time
            for result in post_lookup.results:
                if result.status == TeacherAnnotationStatus.CACHE_HIT and result.labels is not None:
                    results.append(
                        TeacherAnnotationResult(
                            request=result.request,
                            status=TeacherAnnotationStatus.WORKER_RESULT,
                            labels=result.labels,
                        )
                    )
                    resolved_ids.add(result.sample_id)
            remaining = [
                result.request
                for result in post_lookup.results
                if result.status == TeacherAnnotationStatus.CACHE_MISS
                and str(result.request.sample_id) not in resolved_ids
            ]

        cache_write_time = 0.0

        unresolved_requests = [
            request for request in requested if str(request.sample_id) not in resolved_ids
        ]
        unresolved_results = [
            TeacherAnnotationResult(
                request=request,
                status=TeacherAnnotationStatus.UNRESOLVED,
            )
            for request in unresolved_requests
        ]
        results.extend(unresolved_results)

        worker_stats_after = self.worker.snapshot_stats() if self.worker is not None else {}
        stats_delta = (
            self.worker.stats_delta(worker_stats_before, worker_stats_after)
            if self.worker is not None
            else {
                "teacher_batch_size": 0,
                "teacher_batches": 0,
                "batch_fallback_count": 0,
                "oom_retry_count": 0,
                "failed_count": 0,
            }
        )
        annotation_time = time.perf_counter() - ensure_started
        ensure_result = TeacherAnnotationEnsureResult(
            requested_samples=len(requested),
            cache_hits=initial_lookup.cache_hits,
            cache_misses=initial_lookup.cache_misses,
            submitted=submitted,
            waited_sec=waited_sec,
            unresolved_count=len(unresolved_requests),
            annotation_time=annotation_time,
            cache_read_time=cache_read_time,
            cache_write_time=cache_write_time,
            teacher_batch_size=int(stats_delta.get("teacher_batch_size", 0) or 0),
            teacher_batches=int(stats_delta.get("teacher_batches", 0) or 0),
            batch_fallback_count=int(stats_delta.get("batch_fallback_count", 0) or 0),
            oom_retry_count=int(stats_delta.get("oom_retry_count", 0) or 0),
            failed_count=int(stats_delta.get("failed_count", 0) or 0),
            results=results,
            unresolved_requests=unresolved_requests,
        )
        logger.info(
            "[TeacherAnnotation][Ensure] requested={} cache_hits={} cache_misses={} "
            "submitted={} waited_sec={:.3f} unresolved={} "
            "teacher_batch_size={} teacher_batches={} batch_fallback_count={} "
            "oom_retry_count={} failed_count={} annotation_time={:.3f}s "
            "cache_read_time={:.3f}s cache_write_time={:.3f}s",
            ensure_result.requested_samples,
            ensure_result.cache_hits,
            ensure_result.cache_misses,
            ensure_result.submitted,
            ensure_result.waited_sec,
            ensure_result.unresolved_count,
            ensure_result.teacher_batch_size,
            ensure_result.teacher_batches,
            ensure_result.batch_fallback_count,
            ensure_result.oom_retry_count,
            ensure_result.failed_count,
            ensure_result.annotation_time,
            ensure_result.cache_read_time,
            ensure_result.cache_write_time,
        )
        if ensure_result.unresolved_count:
            logger.warning(
                "[TeacherAnnotation][Ensure] unresolved_count={} sample_ids_preview={}",
                ensure_result.unresolved_count,
                ensure_result.unresolved_sample_ids[:10],
            )
        return ensure_result
