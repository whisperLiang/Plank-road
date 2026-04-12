"""Baselines package – unified four-method experiment framework."""

from baselines.base_method import BaseMethod
from baselines.method_factory import create_method
from baselines.metrics import MetricsCollector, DeviceMetrics, OverallMetrics

__all__ = [
    "BaseMethod",
    "create_method",
    "MetricsCollector",
    "DeviceMetrics",
    "OverallMetrics",
]
