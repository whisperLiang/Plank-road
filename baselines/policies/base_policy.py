from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config.baseline import validate_baseline_method


@dataclass(frozen=True)
class BaselineFrameDecision:
    upload_frame: bool
    upload_prediction: bool = True
    request_cloud_inference: bool = False
    is_keyframe: bool = False
    upload_mode: str = "none"
    training_strategy: str = ""
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseBaselinePolicy:
    def __init__(self, method: str, config: object | None = None) -> None:
        self.method = validate_baseline_method(method)
        self.config = config

    @property
    def requires_cloud(self) -> bool:
        return True

    @property
    def frame_filter_enabled(self) -> bool:
        return False

    @property
    def training_strategy(self) -> str:
        return ""

    def decide_frame(self, *, frame_id: int, is_keyframe: bool) -> BaselineFrameDecision:
        raise NotImplementedError
