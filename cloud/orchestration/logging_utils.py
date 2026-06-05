from __future__ import annotations

from cloud.orchestration.fixed_split_dependencies import *  # noqa: F403
from cloud.orchestration.runtime_stage import (
    FIXED_SPLIT_DYNAMIC_BATCH_MAX as _FIXED_SPLIT_DYNAMIC_BATCH_MAX,
    FIXED_SPLIT_DYNAMIC_BATCH_MIN as _FIXED_SPLIT_DYNAMIC_BATCH_MIN,
    cloud_fixed_split_dynamic_batch as _cloud_fixed_split_dynamic_batch,
    cloud_fixed_split_trace_batch_mode as _cloud_fixed_split_trace_batch_mode,
    cloud_fixed_split_trace_batch_size as _cloud_fixed_split_trace_batch_size,
    fixed_split_boundary_from_plan as _fixed_split_boundary_from_plan,
    fixed_split_manifest_has_rebuildable_raw_samples as _fixed_split_manifest_has_rebuildable_raw_samples,
    fixed_split_plan_runtime_contract as _fixed_split_plan_runtime_contract,
    fixed_split_runtime_validation_signature as _fixed_split_runtime_validation_signature,
    fixed_split_validation_batches as _fixed_split_validation_batches,
    negotiate_cached_split_runtime_batch_size as _negotiate_cached_split_runtime_batch_size,
    splitter_dynamic_batch_min as _splitter_dynamic_batch_min,
    splitter_dynamic_batch_range as _splitter_dynamic_batch_range,
)


class StageLoggingMixin:
    @staticmethod
    def _preview_ids(sample_ids: list[str], *, limit: int = 5) -> list[str]:
        return [str(sample_id) for sample_id in sample_ids[: max(0, int(limit))]]


    @staticmethod
    def _log_stage_duration(stage: str, started_at: float) -> float:
        elapsed = time.perf_counter() - started_at
        logger.info("[FixedSplitCL] {} took {:.3f}s.", stage, elapsed)
        return elapsed


    @staticmethod
    def _log_stage_elapsed(stage: str, elapsed: float | None) -> float:
        duration = max(0.0, float(elapsed or 0.0))
        logger.info("[FixedSplitCL] {} took {:.3f}s.", stage, duration)
        return duration

