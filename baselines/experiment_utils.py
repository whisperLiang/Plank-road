"""Shared identifiers and schemas for real baseline experiments."""

from __future__ import annotations

from typing import Mapping

from config.experiment import VALID_METHODS


DISPLAY_NAMES = {
    "plank_road_multi_device": "Plank-road",
    "ekya_style_centralized_scheduling": "Ekya-style",
    "accuracy_trigger_cloud_retraining": "Kong-style",
    "pure_edge_local_updating": "Edge-local",
}


PLANK_ROAD_VARIANTS = {
    "full": {
        "enable_feature_cache": True,
        "enable_split_tail_training": True,
        "enable_resource_aware_trigger": True,
        "enable_feature_upload": True,
    },
    "no_feature_cache": {
        "enable_feature_cache": False,
        "enable_split_tail_training": True,
        "enable_resource_aware_trigger": True,
        "enable_feature_upload": True,
    },
    "no_resource_aware_trigger": {
        "enable_feature_cache": True,
        "enable_split_tail_training": True,
        "enable_resource_aware_trigger": False,
        "enable_feature_upload": True,
    },
    "no_feature_upload": {
        "enable_feature_cache": True,
        "enable_split_tail_training": True,
        "enable_resource_aware_trigger": True,
        "enable_feature_upload": False,
    },
    "no_split_tail": {
        "enable_feature_cache": False,
        "enable_split_tail_training": False,
        "enable_resource_aware_trigger": True,
        "enable_feature_upload": False,
    },
}


def display_name_for_method(method_name: str) -> str:
    return DISPLAY_NAMES.get(method_name, method_name)


def validate_method_name(method_name: str) -> None:
    if method_name not in VALID_METHODS:
        raise ValueError(f"Unknown baseline method {method_name!r}. Valid methods: {VALID_METHODS}")


def normalize_method_variant(method_name: str, method_variant: str | None) -> str:
    variant = str(method_variant or "default")
    if method_name != "plank_road_multi_device":
        return "default"
    return variant


def apply_plank_road_variant(
    config,
    variant: str,
    overrides: Mapping[str, object] | None = None,
) -> None:
    if config.method != "plank_road_multi_device":
        config.method_variant = "default"
        return
    if variant == "default":
        return
    settings = dict(overrides or PLANK_ROAD_VARIANTS.get(variant, {}))
    if not settings:
        raise ValueError(f"Unknown Plank-road method_variant {variant!r}")
    for key, value in settings.items():
        if not hasattr(config.plank_road_multi_device, key):
            continue
        setattr(config.plank_road_multi_device, key, value)
    config.method_variant = variant
