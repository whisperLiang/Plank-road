from __future__ import annotations

from baselines.policies import (
    AccuracyTriggerCloudRetrainingPolicy,
    BaseBaselinePolicy,
    PureEdgeLocalUpdatingPolicy,
)
from config.baseline import ALLOWED_BASELINE_METHODS, validate_baseline_method

_REGISTRY: dict[str, type[BaseBaselinePolicy]] = {
    "pure_edge_local_updating": PureEdgeLocalUpdatingPolicy,
    "accuracy_trigger_cloud_retraining": AccuracyTriggerCloudRetrainingPolicy,
}


def create_policy(method: str, config: object | None = None) -> BaseBaselinePolicy:
    method_name = validate_baseline_method(method)
    return _REGISTRY[method_name](config)


def create_method(config_or_method: object, config: object | None = None) -> BaseBaselinePolicy:
    method_name = str(getattr(config_or_method, "method", config_or_method))
    section = getattr(config_or_method, method_name, config)
    return create_policy(method_name, section)


def registered_methods() -> tuple[str, ...]:
    return ALLOWED_BASELINE_METHODS
