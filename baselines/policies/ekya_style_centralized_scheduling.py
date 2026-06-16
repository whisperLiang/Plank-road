from __future__ import annotations

from baselines.policies.base_policy import BaseBaselinePolicy, BaselineFrameDecision


class EkyaStyleCentralizedSchedulingPolicy(BaseBaselinePolicy):
    def __init__(self, config: object | None = None) -> None:
        super().__init__("ekya_style_centralized_scheduling", config)
        self.upload_raw_frames = bool(getattr(config, "upload_raw_frames", True))
        self.use_frame_filter = bool(getattr(config, "use_frame_filter", False))
        self.cloud_inference = bool(getattr(config, "cloud_inference", True))
        self.return_cloud_inference_to_edge = bool(
            getattr(config, "return_cloud_inference_to_edge", True)
        )
        self.enable_micro_profiling = bool(getattr(config, "enable_micro_profiling", True))
        self.display_source = str(getattr(config, "display_source", "cloud") or "cloud")
        self._training_strategy = str(getattr(config, "training_strategy", "freeze"))

    @property
    def frame_filter_enabled(self) -> bool:
        return self.use_frame_filter

    @property
    def training_strategy(self) -> str:
        return self._training_strategy

    def decide_frame(self, *, frame_id: int, is_keyframe: bool) -> BaselineFrameDecision:
        del is_keyframe
        return BaselineFrameDecision(
            upload_frame=self.upload_raw_frames,
            upload_prediction=False,
            request_cloud_inference=self.cloud_inference,
            is_keyframe=True,
            upload_mode="raw_frame" if self.upload_raw_frames else "none",
            training_strategy=self.training_strategy,
            reason="ekya_raw_frame_upload",
            metadata={
                "frame_id": int(frame_id),
                "return_cloud_inference_to_edge": self.return_cloud_inference_to_edge,
                "display_source": self.display_source,
                "enable_micro_profiling": self.enable_micro_profiling,
            },
        )
