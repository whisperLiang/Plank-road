"""Factory that constructs the correct baseline method from config."""

from __future__ import annotations

from baselines.base_method import BaseMethod
from baselines.accuracy_trigger_cloud_retraining import AccuracyTriggerCloudRetraining
from baselines.ekya_style_centralized_scheduling import EkyaStyleCentralizedScheduling
from baselines.plank_road_multi_device import PlankRoadMultiDevice
from baselines.pure_edge_local_updating import PureEdgeLocalUpdating
from config.experiment import ExperimentConfig


_REGISTRY: dict[str, type[BaseMethod]] = {
    "plank_road_multi_device": PlankRoadMultiDevice,
    "ekya_style_centralized_scheduling": EkyaStyleCentralizedScheduling,
    "accuracy_trigger_cloud_retraining": AccuracyTriggerCloudRetraining,
    "pure_edge_local_updating": PureEdgeLocalUpdating,
}


def create_method(experiment_config: ExperimentConfig) -> BaseMethod:
    """Instantiate the baseline method specified in *experiment_config*.

    Raises ``ValueError`` for unknown method names.
    """
    method_name = experiment_config.method
    if method_name not in _REGISTRY:
        raise ValueError(
            f"Unknown method {method_name!r}. "
            f"Registered methods: {sorted(_REGISTRY.keys())}"
        )
    cls = _REGISTRY[method_name]
    return cls(
        experiment_config=experiment_config,
        num_devices=experiment_config.num_devices,
    )
