from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from cloud.training.types import EarlyStopDecision, ProxyEvalResult


@dataclass
class ProxyEvalConfig:
    enabled: bool = True
    eval_before_retrain: bool = True
    eval_after_first_epoch: bool = True
    eval_final: bool = True
    interval_epochs: int = 10
    max_eval_samples: int | None = 128
    min_delta: float = 0.002
    patience: int = 2
    skip_if_baseline_above: float = 0.98


class ProxyEvalScheduler:
    def __init__(self, config: ProxyEvalConfig) -> None:
        self.config = config

    def should_eval_before_retrain(self) -> bool:
        return bool(self.config.enabled and self.config.eval_before_retrain)

    def should_eval(self, epoch: int, total_epochs: int) -> bool:
        if not self.config.enabled:
            return False
        current_epoch = max(1, int(epoch))
        final_epoch = max(1, int(total_epochs))
        if current_epoch == 1 and self.config.eval_after_first_epoch:
            return True
        interval = int(self.config.interval_epochs or 0)
        if interval > 0 and current_epoch % interval == 0:
            return True
        if self.config.eval_final and current_epoch >= final_epoch:
            return True
        return False


@dataclass
class ProxyEvalHistory:
    results: list[ProxyEvalResult] = field(default_factory=list)
    best_metric: float | None = None
    best_epoch: int | None = None

    def record(self, result: ProxyEvalResult, *, improved: bool) -> None:
        self.results.append(result)
        if improved:
            self.best_metric = result.metric
            self.best_epoch = result.epoch


class ProxyEarlyStopper:
    def __init__(
        self,
        config: ProxyEvalConfig,
        *,
        baseline_metric: float | None = None,
    ) -> None:
        self.config = config
        self.stale_evaluations = 0
        if (
            baseline_metric is not None
            and baseline_metric >= float(config.skip_if_baseline_above)
            and int(config.patience) > 0
        ):
            self.stale_evaluations = max(0, int(config.patience) - 1)

    def record(
        self,
        result: ProxyEvalResult,
        *,
        improved: bool,
        best_metric: float | None,
    ) -> EarlyStopDecision:
        if improved:
            self.stale_evaluations = 0
            return EarlyStopDecision(False, None, self.stale_evaluations)

        self.stale_evaluations += 1
        patience = max(0, int(self.config.patience))
        if patience and self.stale_evaluations >= patience:
            metric_text = (
                "unknown"
                if result.metric is None
                else f"{float(result.metric):.4f}"
            )
            best_text = (
                "unknown"
                if best_metric is None
                else f"{float(best_metric):.4f}"
            )
            reason = (
                f"{self.stale_evaluations} consecutive proxy evaluation(s) "
                f"without >= {float(self.config.min_delta):.6f} improvement "
                f"(latest={metric_text}, best={best_text})"
            )
            return EarlyStopDecision(True, reason, self.stale_evaluations)
        return EarlyStopDecision(False, None, self.stale_evaluations)


def deterministic_proxy_sample_ids(
    gt_annotations: Mapping[object, object],
    max_samples: int | None,
) -> list[str]:
    sample_ids = [str(sample_id) for sample_id in gt_annotations.keys()]
    sample_ids.sort()
    if max_samples is None or int(max_samples) <= 0:
        return sample_ids
    return sample_ids[: int(max_samples)]
