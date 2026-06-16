from __future__ import annotations

from baselines.policies.base_policy import BaseBaselinePolicy, BaselineFrameDecision


class AccuracyTriggerCloudRetrainingPolicy(BaseBaselinePolicy):
    def __init__(self, config: object | None = None) -> None:
        super().__init__("accuracy_trigger_cloud_retraining", config)
        self.reuse_plank_road_frame_filter = bool(
            getattr(config, "reuse_plank_road_frame_filter", True)
        )
        self.upload_keyframes_only = bool(getattr(config, "upload_keyframes_only", True))
        self.trigger_on_cloud_comparison = bool(
            getattr(config, "trigger_on_cloud_comparison", True)
        )
        self.return_model_update = bool(getattr(config, "return_model_update", True))
        self._training_strategy = str(
            getattr(config, "training_strategy", "raw_freeze")
        )

    @property
    def frame_filter_enabled(self) -> bool:
        return self.reuse_plank_road_frame_filter

    @property
    def training_strategy(self) -> str:
        return self._training_strategy

    def decide_frame(self, *, frame_id: int, is_keyframe: bool) -> BaselineFrameDecision:
        upload = bool(is_keyframe or not self.upload_keyframes_only)
        return BaselineFrameDecision(
            upload_frame=upload,
            upload_prediction=upload,
            request_cloud_inference=False,
            is_keyframe=bool(is_keyframe),
            upload_mode="keyframe_raw" if upload else "none",
            training_strategy=self.training_strategy,
            reason="keyframe_selected" if upload else "non_keyframe_filtered",
            metadata={
                "frame_id": int(frame_id),
                "trigger_on_cloud_comparison": self.trigger_on_cloud_comparison,
                "return_model_update": self.return_model_update,
            },
        )
