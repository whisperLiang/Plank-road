from __future__ import annotations

from baselines.policies.base_policy import BaseBaselinePolicy, BaselineFrameDecision


class PureEdgeLocalUpdatingPolicy(BaseBaselinePolicy):
    def __init__(self, config: object | None = None) -> None:
        super().__init__("SURGEON", config)
        self.label_source = str(getattr(config, "label_source", "pseudo_label"))
        self.local_metrics = bool(getattr(config, "local_metrics", True))
        self.upload_metrics_to_cloud = bool(getattr(config, "upload_metrics_to_cloud", False))
        self.upload_frames_to_cloud = bool(getattr(config, "upload_frames_to_cloud", False))
        self.use_cloud_teacher = bool(getattr(config, "use_cloud_teacher", False))
        self._training_strategy = str(getattr(config, "training_strategy", "surgeon_tta"))

    @property
    def requires_cloud(self) -> bool:
        return False

    @property
    def training_strategy(self) -> str:
        return self._training_strategy

    def decide_frame(self, *, frame_id: int, is_keyframe: bool) -> BaselineFrameDecision:
        del frame_id, is_keyframe
        return BaselineFrameDecision(
            upload_frame=False,
            upload_prediction=False,
            is_keyframe=False,
            upload_mode="none",
            training_strategy=self.training_strategy,
            reason="pure_edge_local_only",
            metadata={
                "label_source": self.label_source,
                "local_metrics": self.local_metrics,
                "upload_metrics_to_cloud": self.upload_metrics_to_cloud,
                "upload_frames_to_cloud": self.upload_frames_to_cloud,
                "use_cloud_teacher": self.use_cloud_teacher,
            },
        )
