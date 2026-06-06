from __future__ import annotations

import copy
import time
from typing import Any

import torch
from loguru import logger as default_logger

from cloud.training.proxy_eval import ProxyEarlyStopper, ProxyEvalScheduler
from cloud.training.types import (
    CandidateState,
    FixedSplitTrainingContext,
    FixedSplitTrainingResult,
    ProxyEvalResult,
    ProxyMetrics,
)


class FixedSplitRetrainEngine:
    def run(self, context: FixedSplitTrainingContext) -> FixedSplitTrainingResult:
        plan = context.plan
        log = context.logger or default_logger
        adapter = context.adapter
        config = plan.proxy_eval_config
        scheduler = ProxyEvalScheduler(config)
        total_started = time.perf_counter()
        baseline_state = _snapshot_model_state(context.model)
        proxy_results: list[ProxyEvalResult] = []
        proxy_eval_time = 0.0
        epoch_results = []
        trained_epochs = 0
        early_stop_reason: str | None = None
        can_select_by_proxy = bool(config.enabled and context.gt_annotations)

        log.info(
            "[FixedSplitCL][TrainPlan] model_name={} model_family={} total_samples={} "
            "epochs={} effective_batch_size={} eval_interval={} max_eval_samples={}",
            plan.model_name,
            plan.model_family,
            int(plan.total_samples),
            int(plan.epochs),
            int(plan.effective_batch_size),
            int(config.interval_epochs or 0),
            config.max_eval_samples,
        )

        proxy_metrics_before: ProxyMetrics = {}
        if can_select_by_proxy and context.initial_proxy_metrics is not None:
            proxy_metrics_before = dict(context.initial_proxy_metrics)
            baseline_eval = ProxyEvalResult(
                metrics=dict(proxy_metrics_before),
                metric=adapter.metric_value(proxy_metrics_before),
                elapsed=max(0.0, float(context.initial_proxy_eval_time)),
                epoch=0,
                stage_label="proxy evaluation before retrain",
            )
            proxy_results.append(baseline_eval)
            proxy_eval_time += baseline_eval.elapsed
            self._log_proxy_eval(log, plan, baseline_eval)
        elif can_select_by_proxy and scheduler.should_eval_before_retrain():
            baseline_eval = adapter.evaluate_proxy(
                context,
                epoch=0,
                stage_label="proxy evaluation before retrain",
                max_samples=config.max_eval_samples,
                allow_dead_baseline_fast_path=True,
            )
            proxy_results.append(baseline_eval)
            proxy_eval_time += baseline_eval.elapsed
            proxy_metrics_before = dict(baseline_eval.metrics)
            self._log_proxy_eval(log, plan, baseline_eval)

        baseline_metric = adapter.metric_value(proxy_metrics_before)
        best_candidate: CandidateState | None = None
        if can_select_by_proxy and baseline_metric is not None:
            best_candidate = CandidateState(
                epoch=0,
                state_dict=baseline_state,
                proxy_metrics=dict(proxy_metrics_before),
                proxy_metric=baseline_metric,
                is_baseline=True,
            )
        early_stopper = ProxyEarlyStopper(config, baseline_metric=baseline_metric)
        optimizer = adapter.build_optimizer(context)

        for epoch in range(1, int(plan.epochs) + 1):
            try:
                epoch_result = adapter.train_one_epoch(
                    context,
                    epoch=epoch,
                    total_epochs=int(plan.epochs),
                    optimizer=optimizer,
                )
            except Exception as exc:
                optimizer = self._retry_after_oom_if_possible(
                    context,
                    optimizer=optimizer,
                    best_candidate=best_candidate,
                    baseline_state=baseline_state,
                    exc=exc,
                    log=log,
                )
                epoch_result = adapter.train_one_epoch(
                    context,
                    epoch=epoch,
                    total_epochs=int(plan.epochs),
                    optimizer=optimizer,
                )

            trained_epochs = epoch
            epoch_results.append(epoch_result)
            log.info(
                "[FixedSplitCL][Epoch] model_name={} model_family={} epoch {}/{} "
                "avg_loss={} suffix_forward_backward_time={:.3f}s train_time={:.3f}s",
                plan.model_name,
                plan.model_family,
                int(epoch),
                int(plan.epochs),
                "nan" if epoch_result.loss is None else f"{float(epoch_result.loss):.6f}",
                float(epoch_result.suffix_forward_backward_time),
                float(epoch_result.train_time),
            )

            if not (can_select_by_proxy and scheduler.should_eval(epoch, int(plan.epochs))):
                continue

            proxy_eval = adapter.evaluate_proxy(
                context,
                epoch=epoch,
                stage_label=f"proxy evaluation after epoch {epoch}",
                max_samples=config.max_eval_samples,
            )
            proxy_results.append(proxy_eval)
            proxy_eval_time += proxy_eval.elapsed
            self._log_proxy_eval(log, plan, proxy_eval)

            incumbent_metrics = (
                best_candidate.proxy_metrics if best_candidate is not None else None
            )
            improved = adapter.metrics_are_better(
                proxy_eval.metrics,
                incumbent_metrics,
                min_delta=float(config.min_delta),
            )
            if improved:
                best_candidate = CandidateState(
                    epoch=epoch,
                    state_dict=_snapshot_model_state(context.model),
                    proxy_metrics=dict(proxy_eval.metrics),
                    proxy_metric=proxy_eval.metric,
                    is_baseline=False,
                )
                log.info(
                    "[FixedSplitCL][Candidate] model_name={} model_family={} "
                    "best_epoch={} best_proxy_metric={}",
                    plan.model_name,
                    plan.model_family,
                    int(epoch),
                    _format_metric(proxy_eval.metric),
                )

            decision = early_stopper.record(
                proxy_eval,
                improved=improved,
                best_metric=(
                    best_candidate.proxy_metric if best_candidate is not None else None
                ),
            )
            if decision.should_stop:
                early_stop_reason = decision.reason
                log.info(
                    "[FixedSplitCL][EarlyStop] model_name={} model_family={} epoch={} "
                    "early_stop_reason={}",
                    plan.model_name,
                    plan.model_family,
                    int(epoch),
                    early_stop_reason,
                )
                break

        has_post_training_proxy_eval = any(
            result.epoch is not None and int(result.epoch) > 0
            for result in proxy_results
        )
        proxy_metrics_after = dict(proxy_metrics_before)
        should_restore_best_candidate = (
            can_select_by_proxy
            and best_candidate is not None
            and (not best_candidate.is_baseline or has_post_training_proxy_eval)
        )
        if should_restore_best_candidate and best_candidate is not None:
            if best_candidate.state_dict is not None:
                context.model.load_state_dict(best_candidate.state_dict, strict=False)
                _set_eval(context.model)
            proxy_metrics_after = dict(best_candidate.proxy_metrics or {})
        elif context.gt_annotations and config.enabled and proxy_results:
            proxy_metrics_after = dict(proxy_results[-1].metrics)
        elif (
            context.gt_annotations
            and not proxy_results
            and scheduler.should_eval_before_retrain()
        ):
            final_eval = adapter.evaluate_proxy(
                context,
                epoch=trained_epochs,
                stage_label="proxy evaluation after retrain",
                max_samples=config.max_eval_samples,
            )
            proxy_results.append(final_eval)
            proxy_eval_time += final_eval.elapsed
            proxy_metrics_after = dict(final_eval.metrics)

        external_proxy_eval_time = (
            max(0.0, float(context.initial_proxy_eval_time))
            if context.initial_proxy_metrics is not None
            else 0.0
        )
        total_retraining_time = time.perf_counter() - total_started + external_proxy_eval_time
        suffix_forward_backward_time = sum(
            float(result.suffix_forward_backward_time) for result in epoch_results
        )
        best_epoch = best_candidate.epoch if best_candidate is not None else None
        best_proxy_metric = (
            best_candidate.proxy_metric if best_candidate is not None else None
        )
        log.info(
            "[FixedSplitCL][RetrainProfile] model_name={} model_family={} "
            "total_samples={} epochs={} effective_batch_size={} eval_interval={} "
            "max_eval_samples={} best_epoch={} best_proxy_metric={} trained_epochs={} "
            "early_stop_reason={} suffix_forward_backward_time={:.3f}s "
            "proxy_eval_time={:.3f}s total_retraining_time={:.3f}s",
            plan.model_name,
            plan.model_family,
            int(plan.total_samples),
            int(plan.epochs),
            int(plan.effective_batch_size),
            int(config.interval_epochs or 0),
            config.max_eval_samples,
            best_epoch,
            _format_metric(best_proxy_metric),
            int(trained_epochs),
            early_stop_reason,
            suffix_forward_backward_time,
            proxy_eval_time,
            total_retraining_time,
        )

        return FixedSplitTrainingResult(
            proxy_metrics_before=proxy_metrics_before,
            proxy_metrics_after=proxy_metrics_after,
            baseline_state=baseline_state,
            best_candidate=best_candidate,
            epoch_results=epoch_results,
            proxy_results=proxy_results,
            suffix_forward_backward_time=suffix_forward_backward_time,
            proxy_eval_time=proxy_eval_time,
            total_retraining_time=total_retraining_time,
            best_epoch=best_epoch,
            best_proxy_metric=best_proxy_metric,
            trained_epochs=trained_epochs,
            early_stop_reason=early_stop_reason,
        )

    @staticmethod
    def _log_proxy_eval(
        log: Any,
        plan: Any,
        result: ProxyEvalResult,
    ) -> None:
        log.info(
            "[FixedSplitCL][ProxyEval] model_name={} model_family={} stage={} "
            "epoch={} proxy_metric={} evaluated_samples={} priority_samples={} "
            "random_fill_samples={} elapsed={:.3f}s",
            plan.model_name,
            plan.model_family,
            result.stage_label,
            result.epoch,
            _format_metric(result.metric),
            int(result.metrics.get("evaluated_samples", 0) or 0),
            int(result.metrics.get("priority_gt_samples", 0) or 0),
            int(result.metrics.get("random_fill_gt_samples", 0) or 0),
            float(result.elapsed),
        )

    @staticmethod
    def _retry_after_oom_if_possible(
        context: FixedSplitTrainingContext,
        *,
        optimizer: Any,
        best_candidate: CandidateState | None,
        baseline_state: dict[str, object],
        exc: Exception,
        log: Any,
    ) -> Any:
        del optimizer
        checker = context.is_recoverable_oom
        fallback_batch_size = max(1, int(context.oom_fallback_batch_size))
        if checker is None or not checker(exc):
            raise exc
        current_batch_size = int(context.training_kwargs.get("batch_size") or 0)
        if current_batch_size <= fallback_batch_size:
            raise exc
        restore_state = (
            best_candidate.state_dict
            if best_candidate is not None and best_candidate.state_dict is not None
            else baseline_state
        )
        context.model.load_state_dict(restore_state, strict=False)
        _set_eval(context.model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        context.training_kwargs["batch_size"] = fallback_batch_size
        context.plan.effective_batch_size = fallback_batch_size
        log.warning(
            "[FixedSplitCL][TrainPlan] model_name={} model_family={} CUDA OOM at "
            "batch_size={}; retrying with batch_size={}",
            context.plan.model_name,
            context.plan.model_family,
            current_batch_size,
            fallback_batch_size,
        )
        return context.adapter.build_optimizer(context)


def _snapshot_model_state(model: torch.nn.Module) -> dict[str, object]:
    snapshot: dict[str, object] = {}
    for key, value in model.state_dict().items():
        if torch.is_tensor(value):
            snapshot[key] = value.detach().cpu().clone()
        else:
            snapshot[key] = copy.deepcopy(value)
    return snapshot


def _set_eval(model: torch.nn.Module) -> None:
    model.eval()
    for module in model.modules():
        module.training = False


def _format_metric(metric: float | None) -> str:
    if metric is None:
        return "None"
    return f"{float(metric):.4f}"
