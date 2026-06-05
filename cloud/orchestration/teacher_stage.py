from __future__ import annotations

from collections.abc import Sequence

from loguru import logger

from cloud.annotation import TeacherAnnotationRequest, TeacherAnnotationService


class TeacherAnnotationStage:
    def __init__(
        self,
        service: TeacherAnnotationService,
        *,
        wait_timeout_sec: float,
    ) -> None:
        self.service = service
        self.wait_timeout_sec = float(wait_timeout_sec)

    def ensure_low_quality(
        self,
        requests: Sequence[TeacherAnnotationRequest],
    ) -> dict[str, dict[str, object]]:
        ensure_result = self.service.ensure_many(
            list(requests),
            wait=True,
            timeout_sec=self.wait_timeout_sec,
        )
        if ensure_result.unresolved_count:
            logger.warning(
                "[TeacherAnnotation][Ensure] deferring unresolved low-quality samples before canonical staging: "
                "unresolved_count={} sample_ids_preview={}",
                ensure_result.unresolved_count,
                ensure_result.unresolved_sample_ids[:10],
            )
        return {
            str(sample_id): dict(labels)
            for sample_id, labels in ensure_result.labels_by_sample_id.items()
        }
