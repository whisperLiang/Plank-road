from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cloud.orchestration.results import SampleRebuildResult
from cloud.sample_pool import CloudSamplePool
from model_management.split_contract import SplitRuntimeContract


class CanonicalSampleStage:
    def __init__(self, sample_pool: CloudSamplePool) -> None:
        self.sample_pool = sample_pool

    def rebuild(
        self,
        *,
        split_contract: SplitRuntimeContract,
        existing_active: list[Mapping[str, Any]],
        pending_high_quality: list[Mapping[str, Any]],
        new_low_quality: list[Mapping[str, Any]],
    ) -> SampleRebuildResult:
        rebuild_stats, kept_records = self.sample_pool.rebuild_canonical_training_pool(
            split_contract=split_contract,
            existing_active_samples=existing_active,
            pending_high_quality_samples=pending_high_quality,
            new_low_quality_samples=new_low_quality,
        )
        return SampleRebuildResult(
            rebuild_stats=dict(rebuild_stats),
            kept_records=list(kept_records),
            existing_active=[dict(sample) for sample in existing_active],
            pending_high_quality=[dict(sample) for sample in pending_high_quality],
            staging_low_quality=[dict(sample) for sample in new_low_quality],
        )
