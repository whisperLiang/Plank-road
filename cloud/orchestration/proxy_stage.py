from __future__ import annotations

from cloud.training import FixedSplitProxyDecision, FixedSplitProxyEvaluator

__all__ = ["FixedSplitProxyDecision", "FixedSplitProxyEvaluator"]

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


class ProxyStageMixin:
    def _proxy_eval_frame_cache(self) -> dict[str, np.ndarray | None] | None:
        return self._fixed_split_proxy_evaluator().new_frame_cache()


    def _fixed_split_proxy_evaluator(self) -> FixedSplitProxyEvaluator:
        return FixedSplitProxyEvaluator(
            device=self.device,
            default_batch_size=self.batch_size,
            max_samples=self.proxy_eval_max_samples,
            frame_cache_enabled=self.proxy_eval_frame_cache_enabled,
            tinynext_threshold_candidates=self.proxy_eval_threshold_candidates,
        )


    def _evaluate_fixed_split_proxy_map(
        self,
        model: torch.nn.Module,
        *,
        frame_dir: str,
        gt_annotations: Mapping[str, Mapping[str, object]],
        model_name: str,
        sample_metadata_by_id: Mapping[str, Mapping[str, object]] | None = None,
        frame_cache: dict[str, np.ndarray | None] | None = None,
        max_samples: int | None = None,
        inference_batch_size: int | None = None,
        split_cache_path: str | None = None,
        splitter: UniversalModelSplitter | None = None,
        split_candidate=None,
        preloaded_records: Mapping[object, Mapping[str, object]] | None = None,
        proxy_cache_threshold_low: float | None = None,
    ) -> dict[str, float | int | None]:
        return self._fixed_split_proxy_evaluator().evaluate_detection(
            model,
            frame_dir=frame_dir,
            gt_annotations=gt_annotations,
            model_name=model_name,
            sample_metadata_by_id=sample_metadata_by_id,
            frame_cache=frame_cache,
            max_samples=max_samples,
            inference_batch_size=inference_batch_size,
            split_cache_path=split_cache_path,
            splitter=splitter,
            split_candidate=split_candidate,
            preloaded_records=preloaded_records,
            proxy_cache_threshold_low=proxy_cache_threshold_low,
        )


    def _evaluate_tinynext_proxy_map(
        self,
        model: torch.nn.Module,
        *,
        frame_dir: str,
        gt_annotations: Mapping[str, Mapping[str, object]],
        model_name: str,
        sample_metadata_by_id: Mapping[str, Mapping[str, object]] | None = None,
        frame_cache: dict[str, np.ndarray | None] | None = None,
        max_samples: int | None = None,
        candidate_thresholds: list[float] | None = None,
        inference_batch_size: int | None = None,
        stage_label: str,
        split_cache_path: str | None = None,
        splitter: UniversalModelSplitter | None = None,
        split_candidate=None,
        preloaded_records: Mapping[object, Mapping[str, object]] | None = None,
        allow_dead_baseline_fast_path: bool = False,
    ) -> dict[str, float | int | None]:
        return self._fixed_split_proxy_evaluator().evaluate_tinynext(
            model,
            frame_dir=frame_dir,
            gt_annotations=gt_annotations,
            model_name=model_name,
            sample_metadata_by_id=sample_metadata_by_id,
            frame_cache=frame_cache,
            max_samples=max_samples,
            candidate_thresholds=candidate_thresholds,
            inference_batch_size=inference_batch_size,
            stage_label=stage_label,
            split_cache_path=split_cache_path,
            splitter=splitter,
            split_candidate=split_candidate,
            preloaded_records=preloaded_records,
            allow_dead_baseline_fast_path=allow_dead_baseline_fast_path,
            logger=logger,
        )
