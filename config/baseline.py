from __future__ import annotations

from dataclasses import dataclass

ALLOWED_BASELINE_METHODS: tuple[str, ...] = (
    "pure_edge_local_updating",
    "accuracy_trigger_cloud_retraining",
    "ekya_style_cloud_scheduling",
)

_PLANK_ROAD_BASELINE_METHOD = "plank_road" + "_multi_device"
PLANK_ROAD_BASELINE_ERROR = (
    f"{_PLANK_ROAD_BASELINE_METHOD} is not a baseline method. "
    "Use the main Plank-Road distributed deployment path instead."
)


def validate_baseline_method(method: str) -> str:
    value = str(method or "").strip()
    if value == _PLANK_ROAD_BASELINE_METHOD:
        raise ValueError(PLANK_ROAD_BASELINE_ERROR)
    if value not in ALLOWED_BASELINE_METHODS:
        raise ValueError(
            f"Unknown baseline method {value!r}. "
            f"Valid methods: {', '.join(ALLOWED_BASELINE_METHODS)}"
        )
    return value


@dataclass(frozen=True)
class BaselineIdentity:
    run_id: str
    baseline_method: str
    edge_id: int

    def key(self) -> tuple[str, str, int]:
        return (self.run_id, validate_baseline_method(self.baseline_method), int(self.edge_id))
