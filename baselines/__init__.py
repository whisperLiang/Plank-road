"""Real-execution baseline framework for video continual learning."""

from baselines.base_method import BaseMethod
from baselines.method_factory import create_method
from baselines.metrics import DeviceMetrics, MetricsCollector, OverallMetrics

__all__ = [
    "BaseMethod",
    "create_method",
    "MetricsCollector",
    "DeviceMetrics",
    "OverallMetrics",
]
