from __future__ import annotations

from cloud.orchestration.fixed_split_dependencies import *  # noqa: F403
from cloud.orchestration.checkpoint_stage import CheckpointStageMixin
from cloud.orchestration.logging_utils import StageLoggingMixin
from cloud.orchestration.proxy_stage import ProxyStageMixin
from cloud.orchestration.request_context import RequestContextMixin
from cloud.orchestration.runtime_stage import (
    FIXED_SPLIT_DYNAMIC_BATCH_MIN as _FIXED_SPLIT_DYNAMIC_BATCH_MIN,
    FixedSplitRuntimeContractMixin,
    splitter_dynamic_batch_min as _splitter_dynamic_batch_min,
)
from cloud.orchestration.runtime_template_stage import FixedSplitRuntimeTemplateMixin
from cloud.orchestration.sample_stage import CanonicalSampleStage, SampleStageMixin
from cloud.orchestration.settings import PipelineLifecycleMixin
from cloud.orchestration.teacher_stage import TeacherAnnotationMixin
from cloud.orchestration.training_stage import TrainingStageMixin


# ---------------------------------------------------------------------------
# Cloud-side Continual Learning
# ---------------------------------------------------------------------------

class FixedSplitPipeline(
    PipelineLifecycleMixin,
    RequestContextMixin,
    StageLoggingMixin,
    CheckpointStageMixin,
    TeacherAnnotationMixin,
    FixedSplitRuntimeContractMixin,
    FixedSplitRuntimeTemplateMixin,
    SampleStageMixin,
    ProxyStageMixin,
    TrainingStageMixin,
):
    """Performs ground-truth labelling and model retraining on the cloud side.

    Workflow triggered when the edge detects drift:
      1. Edge sends selected frame indices and the path of its local cache.
      2. Cloud runs the large model on each frame to obtain ground-truth boxes.
      3. Cloud saves a CSV annotation file inside the cache directory.
      4. Cloud retrains a **fresh copy** of the lightweight edge model.
      5. Cloud returns the updated state-dict bytes (base-64 encoded).

    The edge model weights are kept separately from the cloud inference model.
    """

    def get_ground_truth_and_retrain(
        self,
        edge_id: int,
        frame_indices: list[int],
        cache_path: str,
    ) -> tuple[bool, str, str]:
        del edge_id, frame_indices, cache_path
        message = "fixed-split training failed; legacy full-image retrain has been removed"
        logger.error("[CL] {}", message)
        return False, "", message

    def get_ground_truth_and_fixed_split_retrain(
        self,
        edge_id: int,
        bundle_cache_path: str,
        *,
        num_epoch: int | None = None,
    ) -> tuple[bool, str, str]:
        effective_num_epoch = self.default_num_epoch if num_epoch is None else int(num_epoch)

        with self._training_job_scope(edge_id):
            total_round_started = time.perf_counter()
            try:
                stage_started = time.perf_counter()
                materialized_manifest = materialize_low_quality_trigger_bundle(
                    bundle_cache_path,
                    feature_store=self._feature_readiness_service().store(),
                )
                if materialized_manifest is None:
                    raise RuntimeError(
                        "Shard-based continual-learning trigger payload must contain "
                        "trigger_manifest.json; legacy bundle_manifest.json uploads are no longer supported."
                    )
                manifest = materialized_manifest
                self._log_stage_duration("loading bundle manifest", stage_started)
                if manifest.get("protocol_version") != LOW_QUALITY_TRIGGER_PROTOCOL_VERSION:
                    raise RuntimeError(
                        f"Unexpected bundle protocol version: {manifest.get('protocol_version')!r}"
                    )
                current_model_name = self._resolve_fixed_split_model_name(manifest)
                manifest_model_metadata = _manifest_model_metadata(manifest)
                manifest_runtime_input_shape = _runtime_input_tensor_shape_from_metadata(
                    manifest
                )
                early_teacher_requests = self._build_low_quality_raw_teacher_annotation_requests(
                    bundle_cache_path=bundle_cache_path,
                    manifest=manifest,
                    edge_id=edge_id,
                    model_id=current_model_name,
                    target_model_metadata=manifest_model_metadata,
                )
                self._submit_low_quality_teacher_annotations(early_teacher_requests)
                bundle_sample_count = self._count_manifest_training_samples(manifest)
                effective_batch_size = self._resolve_fixed_split_runtime_batch_size(
                    current_model_name,
                    num_train_samples=bundle_sample_count,
                )
                bundle_model_version = _normalize_model_version(
                    manifest.get("model", {}).get("model_version", "0"),
                    field_name="bundle model version",
                )
                sample_pool = self._cloud_sample_pool_for_manifest(
                    edge_id=edge_id,
                    manifest=manifest,
                )
                sample_pool = self._reset_initial_cloud_state_if_needed(
                    edge_id=edge_id,
                    manifest=manifest,
                    model_name=current_model_name,
                    sample_pool=sample_pool,
                    fallback_model_version=bundle_model_version,
                    allow_without_session=True,
                )
                next_checkpoint_model_version = _increment_model_version(
                    bundle_model_version,
                    field_name="bundle model version",
                )
                baseline_source = f"native {self._native_training_source_label(current_model_name)}"
                existing_contract = self._load_split_runtime_contract(
                    edge_id=edge_id,
                    manifest=manifest,
                )
                front_version = str(
                    self._sample_pool_manifest_context(manifest).get("front_version") or "0"
                )
                if (
                    existing_contract is None
                    and front_version == "0"
                    and bundle_model_version != "0"
                ):
                    message = (
                        "Missing SplitRuntimeContract for front_version=0; refusing "
                        f"to create it from tail checkpoint model_version={bundle_model_version}. "
                        "Run a model_version=0 training job first so the contract is "
                        "created from native pretrained front weights."
                    )
                    logger.warning("[FixedSplitCL] {}", message)
                    self._log_stage_duration("total round time", total_round_started)
                    return False, "", message

                if bundle_model_version == "0":
                    logger.info(
                        "[FixedSplitCL] Bundle model_version=0 for edge {}; ignoring any cached {} weights and starting from native {} weights.",
                        edge_id,
                        current_model_name,
                        self._native_training_source_label(current_model_name),
                    )
                    tmp_model = self._load_edge_training_model(
                        model_name=current_model_name,
                        edge_id=edge_id,
                        cache_policy="native_only",
                        runtime_input_tensor_shape=manifest_runtime_input_shape,
                        model_metadata=manifest_model_metadata,
                    )
                else:
                    metadata = self._require_matching_edge_weights_metadata(
                        model_name=current_model_name,
                        edge_id=edge_id,
                        bundle_model_version=bundle_model_version,
                    )
                    logger.info(
                        "[FixedSplitCL] Resuming edge {} {} training from persisted checkpoint version {}.",
                        edge_id,
                        current_model_name,
                        metadata["checkpoint_model_version"],
                    )
                    baseline_source = "edge-scoped cache"
                    tmp_model = self._load_edge_training_model(
                        model_name=current_model_name,
                        edge_id=edge_id,
                        cache_policy="edge_only",
                        runtime_input_tensor_shape=manifest_runtime_input_shape,
                        model_metadata=manifest_model_metadata,
                    )
                weights_metadata = self._build_checkpoint_weights_metadata(
                    edge_id=edge_id,
                    model_name=current_model_name,
                    model=tmp_model,
                    checkpoint_model_version=next_checkpoint_model_version,
                    source_base_model_version=bundle_model_version,
                    runtime_input_tensor_shape=manifest_runtime_input_shape,
                )
                stage_started = time.perf_counter()
                prepared_splitter, prepared_candidate = self._build_bundle_splitter(
                    tmp_model,
                    manifest,
                    bundle_root=bundle_cache_path,
                    runtime_batch_size=effective_batch_size,
                )
                prepared_trace_sample_input = None
                self._log_stage_duration("runtime template load / bind", stage_started)
                pool_runtime_input_tensor_shape = self._infer_pool_runtime_input_tensor_shape(
                    tmp_model,
                    bundle_root=bundle_cache_path,
                    manifest=manifest,
                    prepared_trace_sample_input=prepared_trace_sample_input,
                )
                pool_model_input_size = (
                    (
                        int(pool_runtime_input_tensor_shape[-2]),
                        int(pool_runtime_input_tensor_shape[-1]),
                    )
                    if pool_runtime_input_tensor_shape is not None
                    and len(pool_runtime_input_tensor_shape) >= 3
                    else None
                )
                pool_input_resize_mode = (
                    get_split_runtime_input_resize_mode(get_split_runtime_model(tmp_model))
                    or "direct_resize"
                )
                stage_started = time.perf_counter()
                teacher_requests = self._build_low_quality_raw_teacher_annotation_requests(
                    bundle_cache_path=bundle_cache_path,
                    manifest=manifest,
                    edge_id=edge_id,
                    model_id=current_model_name,
                    target_model_metadata=manifest_model_metadata,
                )
                gt_annotations = self._teacher_annotation_stage().ensure_low_quality(
                    teacher_requests,
                )
                current_low_quality_gt_sample_ids = {
                    str(sample_id) for sample_id in gt_annotations.keys()
                }
                self._log_stage_duration("teacher annotation ensure", stage_started)
                pending_high_quality = sample_pool.load_pending_high_quality_samples()
                contract_layout_tensors = self._contract_layout_tensors_from_runtime(
                    splitter=prepared_splitter,
                    candidate=prepared_candidate,
                    input_tensor_shape=[
                        int(dim)
                        for dim in list(
                            self._sample_pool_manifest_context(manifest).get(
                                "input_tensor_shape",
                                [],
                            )
                            or []
                        )
                    ],
                )
                split_contract = self._get_or_create_split_runtime_contract(
                    edge_id=edge_id,
                    manifest=manifest,
                    feature_tensors=contract_layout_tensors,
                    contract_layout_tensors=contract_layout_tensors,
                    model=tmp_model,
                    splitter=prepared_splitter,
                    candidate=prepared_candidate,
                    bundle_root=bundle_cache_path,
                    create_if_missing=True,
                )
                self._log_pending_high_quality_layout_alignment(
                    pending_high_quality=pending_high_quality,
                    split_contract=split_contract,
                    expected_source="runtime",
                    low_quality_tensors=None,
                )
                stage_started = time.perf_counter()
                low_quality_feature_entries = self._prepare_low_quality_feature_entries(
                    tmp_model,
                    manifest,
                    bundle_cache_path=bundle_cache_path,
                    gt_annotations=gt_annotations,
                    split_contract=split_contract,
                    splitter=prepared_splitter,
                    candidate=prepared_candidate,
                    model_name=current_model_name,
                    runtime_batch_size=effective_batch_size,
                )
                low_quality_staging_candidates = self._build_low_quality_staging_candidates(
                    feature_entries=low_quality_feature_entries,
                    model_input_size=pool_model_input_size,
                    resize_mode=pool_input_resize_mode,
                )
                staging_stats = sample_pool.stage_low_quality_samples(
                    low_quality_staging_candidates
                )
                staging_low_quality = sample_pool.load_staging_low_quality_samples()
                existing_active = sample_pool.load_active_samples_for_rebuild(
                    split_contract=split_contract,
                )
                rebuild_result = CanonicalSampleStage(sample_pool).rebuild(
                    split_contract=split_contract,
                    existing_active=existing_active,
                    pending_high_quality=pending_high_quality,
                    new_low_quality=staging_low_quality,
                )
                rebuild_stats = rebuild_result.rebuild_stats
                kept_records = rebuild_result.kept_records
                self._log_stage_duration("feature readiness + canonical sample-pool rebuild", stage_started)
                self._log_sample_rebuild_summary(
                    split_contract=split_contract,
                    existing_active=existing_active,
                    pending_high_quality=pending_high_quality,
                    staging_low_quality=staging_low_quality,
                    rebuild_stats=rebuild_stats,
                    staging_stats=staging_stats,
                    low_quality_candidate_count=len(low_quality_staging_candidates),
                )

                stage_started = time.perf_counter()
                (
                    bundle_info,
                    frame_dir,
                    preloaded_records,
                    gt_annotations,
                    sample_metadata_by_id,
                    _training_view,
                    _training_view_stats,
                ) = self._build_training_cache_view_from_canonical_active(
                    sample_pool,
                    contract=split_contract,
                    model_name=current_model_name,
                    edge_id=edge_id,
                )
                self._log_stage_duration("training cache view materialization", stage_started)
                active_sample_count = len(bundle_info["all_sample_ids"])
                required_dynamic_batch_min = max(
                    _FIXED_SPLIT_DYNAMIC_BATCH_MIN,
                    _splitter_dynamic_batch_min(prepared_splitter),
                )
                if active_sample_count < required_dynamic_batch_min:
                    message = (
                        "Not enough compatible samples for dynamic batch runtime: "
                        f"active_samples={active_sample_count}, "
                        f"required_min={required_dynamic_batch_min}."
                    )
                    logger.warning("[FixedSplitCL] {}", message)
                    self._log_stage_duration("total round time", total_round_started)
                    return False, "", message
                effective_batch_size = self._resolve_fixed_split_runtime_batch_size(
                    current_model_name,
                    num_train_samples=active_sample_count,
                )
                training_cache_path = str(bundle_info.get("training_view_path") or "")
                proxy_sample_random_seed = str(
                    bundle_info.get("training_view_id")
                    or bundle_info.get("generation_id")
                    or f"{edge_id}:{current_model_name}"
                )
                effective_batch_size = self._negotiate_cached_split_runtime_batch_size(
                    current_model_name=current_model_name,
                    training_cache_path=training_cache_path,
                    all_sample_ids=bundle_info["all_sample_ids"],
                    gt_annotations=gt_annotations,
                    prepared_splitter=prepared_splitter,
                    prepared_candidate=prepared_candidate,
                    configured_batch_size=effective_batch_size,
                    preloaded_records=preloaded_records,
                    manifest=manifest,
                )
                logger.info(
                    "[FixedSplitCL] Training from {} canonical-active sample(s) via TrainingCacheView(source=canonical_active) ({} label entry/entries).",
                    len(bundle_info["all_sample_ids"]),
                    len(gt_annotations),
                )
                if self.connectivity_smoke_only:
                    stage_started = time.perf_counter()
                    encoded = base64.b64encode(
                        self._serialise_model_bytes(
                            tmp_model,
                            model_name=current_model_name,
                            edge_id=edge_id,
                            weights_metadata=weights_metadata,
                        )
                    ).decode("utf-8")
                    self._log_stage_duration("serialization / encoding", stage_started)
                    self._log_stage_duration("total round time", total_round_started)
                    logger.success(
                        "[FixedSplitCL][ConnectivitySmoke] Connected edge inference, "
                        "sample upload, cloud annotation, feature rebuild, contract "
                        "creation, and tail-runtime preparation for edge {} with {} "
                        "active sample(s) ({} GT-annotated); skipped full retraining.",
                        edge_id,
                        len(bundle_info["all_sample_ids"]),
                        len(gt_annotations),
                    )
                    return (
                        True,
                        encoded,
                        "Fixed split connectivity smoke successful; skipped full retraining",
                    )
                proxy_evaluator = self._fixed_split_proxy_evaluator()
                proxy_eval_frame_cache = proxy_evaluator.new_frame_cache()

                stage_started = time.perf_counter()
                if gt_annotations and model_zoo.get_model_family(current_model_name) == "tinynext":
                    proxy_metrics_before = self._evaluate_tinynext_proxy_map(
                        tmp_model,
                        frame_dir=frame_dir,
                        gt_annotations=gt_annotations,
                        model_name=current_model_name,
                        sample_metadata_by_id=sample_metadata_by_id,
                        frame_cache=proxy_eval_frame_cache,
                        max_samples=self.proxy_eval_max_samples,
                        candidate_thresholds=self.proxy_eval_threshold_candidates,
                        inference_batch_size=effective_batch_size,
                        stage_label="proxy evaluation before retrain",
                        split_cache_path=training_cache_path,
                        splitter=prepared_splitter,
                        split_candidate=prepared_candidate,
                        preloaded_records=preloaded_records,
                        allow_dead_baseline_fast_path=True,
                        priority_sample_ids=current_low_quality_gt_sample_ids,
                        random_fill_seed=proxy_sample_random_seed,
                    )
                else:
                    proxy_metrics_before = self._evaluate_fixed_split_proxy_map(
                        tmp_model,
                        frame_dir=frame_dir,
                        gt_annotations=gt_annotations,
                        model_name=current_model_name,
                        sample_metadata_by_id=sample_metadata_by_id,
                        frame_cache=proxy_eval_frame_cache,
                        max_samples=self.proxy_eval_max_samples,
                        inference_batch_size=effective_batch_size,
                        split_cache_path=training_cache_path,
                        splitter=prepared_splitter,
                        split_candidate=prepared_candidate,
                        preloaded_records=preloaded_records,
                        priority_sample_ids=current_low_quality_gt_sample_ids,
                        random_fill_seed=proxy_sample_random_seed,
                    )
                proxy_metrics_before_elapsed = time.perf_counter() - stage_started
                self._log_stage_duration("proxy evaluation before retrain", stage_started)
                is_wrapper_fixed_split = bool(model_zoo.is_wrapper_model(current_model_name))

                if (
                    gt_annotations
                    and is_wrapper_fixed_split
                    and proxy_evaluator.metrics_indicate_dead_detector(proxy_metrics_before)
                    and bool(bundle_info.get("from_sample_pool", False))
                ):
                    logger.warning(
                        "[FixedSplitCL] Cached {} weights produced no detections on {} pool label sample(s), "
                        "but keeping the cached model because resetting would invalidate active cloud sample-pool features.",
                        current_model_name,
                        len(gt_annotations),
                    )

                proxy_metrics_after, baseline_state = self._run_fixed_split_retrain(
                    tmp_model,
                    current_model_name=current_model_name,
                    bundle_info=bundle_info,
                    manifest=manifest,
                    bundle_cache_path=bundle_cache_path,
                    training_cache_path=training_cache_path,
                    frame_dir=frame_dir,
                    gt_annotations=gt_annotations,
                    num_epoch=effective_num_epoch,
                    proxy_metrics_before=proxy_metrics_before,
                    proxy_metrics_before_elapsed=proxy_metrics_before_elapsed,
                    prepared_trace_sample_input=prepared_trace_sample_input,
                    prepared_splitter=prepared_splitter,
                    prepared_candidate=prepared_candidate,
                    effective_batch_size=effective_batch_size,
                    sample_metadata_by_id=sample_metadata_by_id,
                    proxy_eval_frame_cache=proxy_eval_frame_cache,
                    preloaded_records=preloaded_records,
                    proxy_priority_sample_ids=current_low_quality_gt_sample_ids,
                    proxy_sample_random_seed=proxy_sample_random_seed,
                )
                proxy_summary = proxy_evaluator.format_summary(
                    proxy_metrics_before,
                    proxy_metrics_after,
                )
                if proxy_evaluator.metrics_skipped_full_proxy(proxy_metrics_before):
                    logger.info(
                        "[FixedSplitCL] Initial TinyNeXt baseline proxy_mAP@0.5 used "
                        "{}-sample dead-baseline subset fast path; final candidate was "
                        "evaluated on {} sample(s).",
                        int(proxy_metrics_before.get("subset_proxy_sample_count", 0) or 0),
                        int(proxy_metrics_after.get("evaluated_samples", 0) or 0),
                    )
                if proxy_summary is not None:
                    logger.info("[FixedSplitCL] {}", proxy_summary)
                else:
                    logger.info(
                        "[FixedSplitCL] Proxy mAP skipped "
                        "(gt_samples={}, evaluated={}, empty_gt={}, missing_frame={}).",
                        int(proxy_metrics_after.get("total_gt_samples", 0)),
                        int(proxy_metrics_after.get("evaluated_samples", 0)),
                        int(proxy_metrics_after.get("skipped_empty_gt", 0)),
                        int(proxy_metrics_after.get("skipped_missing_frame", 0)),
                    )

                if proxy_evaluator.metrics_skipped_full_proxy(proxy_metrics_before):
                    logger.info(
                        "[FixedSplitCL] Rechecking full TinyNeXt baseline proxy before final "
                        "candidate decision because the initial baseline used the subset fast path."
                    )
                    candidate_state = proxy_evaluator.snapshot_model_state(tmp_model)
                    proxy_evaluator.restore_model_state(tmp_model, baseline_state)
                    full_baseline_metrics = self._evaluate_tinynext_proxy_map(
                        tmp_model,
                        frame_dir=frame_dir,
                        gt_annotations=gt_annotations,
                        model_name=current_model_name,
                        sample_metadata_by_id=sample_metadata_by_id,
                        frame_cache=proxy_eval_frame_cache,
                        max_samples=self.proxy_eval_max_samples,
                        candidate_thresholds=self.proxy_eval_threshold_candidates,
                        inference_batch_size=effective_batch_size,
                        stage_label="full baseline proxy recheck",
                        split_cache_path=training_cache_path,
                        splitter=prepared_splitter,
                        split_candidate=prepared_candidate,
                        preloaded_records=preloaded_records,
                        priority_sample_ids=current_low_quality_gt_sample_ids,
                        random_fill_seed=proxy_sample_random_seed,
                    )
                    proxy_evaluator.restore_model_state(tmp_model, candidate_state)
                    proxy_metrics_before = full_baseline_metrics
                    proxy_summary = proxy_evaluator.format_summary(
                        proxy_metrics_before,
                        proxy_metrics_after,
                    )
                    if proxy_summary is not None:
                        logger.info("[FixedSplitCL] {}", proxy_summary)

                rejection_reason = proxy_evaluator.rejection_reason(
                    proxy_metrics_before,
                    proxy_metrics_after,
                )
                if rejection_reason is not None:
                    logger.warning(
                        "[FixedSplitCL] Rejecting retrained {} weights for edge {}: {}",
                        current_model_name,
                        edge_id,
                        rejection_reason,
                    )
                    proxy_evaluator.restore_model_state(tmp_model, baseline_state)
                    stage_started = time.perf_counter()
                    encoded = base64.b64encode(
                        self._serialise_model_bytes(
                            tmp_model,
                            model_name=current_model_name,
                            edge_id=edge_id,
                            weights_metadata=weights_metadata,
                        )
                    ).decode("utf-8")
                    self._log_stage_duration("serialization / encoding", stage_started)
                    self._log_stage_duration("total round time", total_round_started)
                    fallback_message = (
                        f"Kept {baseline_source} weights; rejected retrained weights because {rejection_reason}"
                    )
                    if proxy_summary is not None:
                        fallback_message = f"{fallback_message}; {proxy_summary}"
                    else:
                        fallback_message = f"{fallback_message}; proxy_mAP@0.5 skipped"
                    return True, encoded, fallback_message

                stage_started = time.perf_counter()
                encoded = base64.b64encode(
                    self._serialise_model_bytes(
                        tmp_model,
                        model_name=current_model_name,
                        edge_id=edge_id,
                        weights_metadata=weights_metadata,
                    )
                ).decode("utf-8")
                self._log_stage_duration("serialization / encoding", stage_started)
                self._log_stage_duration("total round time", total_round_started)
                success_message = (
                    f"Fixed split retraining successful; {proxy_summary}"
                    if proxy_summary is not None
                    else "Fixed split retraining successful; proxy_mAP@0.5 skipped"
                )
                logger.success(
                    "[FixedSplitCL] {} done for edge {} with {} samples ({} GT-annotated).",
                    "Retraining",
                    edge_id,
                    len(bundle_info["all_sample_ids"]),
                    len(gt_annotations),
                )
                return True, encoded, success_message
            except Exception as exc:
                self._log_stage_duration("total round time", total_round_started)
                message = "fixed-split training failed; legacy full-image retrain has been removed"
                logger.exception("[FixedSplitCL] {} for edge {}: {}", message, edge_id, exc)
                return False, "", f"{message}: {exc}"
