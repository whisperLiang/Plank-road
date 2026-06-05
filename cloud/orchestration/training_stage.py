from __future__ import annotations

from cloud.training import FixedSplitRetrainEngine, FixedSplitTrainingContext
from cloud.training.types import FixedSplitTrainingResult


class FixedSplitTrainingStage:
    def __init__(self, engine: FixedSplitRetrainEngine | None = None) -> None:
        self.engine = engine or FixedSplitRetrainEngine()

    def run(self, context: FixedSplitTrainingContext) -> FixedSplitTrainingResult:
        return self.engine.run(context)
