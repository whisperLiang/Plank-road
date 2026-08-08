from __future__ import annotations

from baselines.policies import (
    AccuracyTriggerCloudRetrainingPolicy,
    BaseBaselinePolicy,
    PureEdgeLocalUpdatingPolicy,
)
from config.baseline import (
    ALLOWED_BASELINE_METHODS,
    CATR_METHOD,
    SURGEON_METHOD,
    validate_baseline_method,
)

_REGISTRY: dict[str, type[BaseBaselinePolicy]] = {
    SURGEON_METHOD: PureEdgeLocalUpdatingPolicy,
    CATR_METHOD: AccuracyTriggerCloudRetrainingPolicy,
}


def create_policy(method: str, config: object | None = None) -> BaseBaselinePolicy:
    method_name = validate_baseline_method(method)
    if method_name not in _REGISTRY:
        raise ValueError(
            f"Baseline method {method_name!r} does not use the edge policy factory."
        )
    return _REGISTRY[method_name](config)


def create_method(config_or_method: object, config: object | None = None) -> BaseBaselinePolicy:
    method_name = str(getattr(config_or_method, "method", config_or_method))
    section = getattr(config_or_method, method_name, config)
    return create_policy(method_name, section)


def registered_methods() -> tuple[str, ...]:
    return ALLOWED_BASELINE_METHODS
