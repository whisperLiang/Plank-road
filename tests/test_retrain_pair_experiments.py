from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools import run_retrain_pair_experiments as experiments


def _base_config():
    return SimpleNamespace(
        client=SimpleNamespace(
            lightweight="tinynext_s",
            drift_detection=SimpleNamespace(confidence_threshold=0.8),
            split_learning=SimpleNamespace(
                fixed_split=SimpleNamespace(),
            ),
        ),
        server=SimpleNamespace(
            edge_model_name="tinynext_s",
            golden="rtdetr_x",
            continual_learning=SimpleNamespace(num_epoch=2),
        ),
    )


def test_run_pair_experiment_returns_failure_when_split_plan_fails(tmp_path, monkeypatch):
    frames = [np.zeros((8, 8, 3), dtype=np.uint8)]
    frame_indices = [300]

    monkeypatch.setattr(experiments, "_resolve_local_weights_path", lambda *args, **kwargs: "dummy-weights")
    monkeypatch.setattr(experiments, "_sample_video_frames", lambda *args, **kwargs: (frame_indices, frames))

    class DummyObjectDetection:
        def __init__(self, config, type):
            self.model_name = getattr(config, "lightweight", getattr(config, "golden", "dummy"))
            self.model = torch.nn.Linear(1, 1)

    class DummyLearner:
        def __init__(self, config, large_object_detection):
            self.config = config
            self.large_object_detection = large_object_detection
            self.weight_folder = ""

    monkeypatch.setattr(experiments, "Object_Detection", DummyObjectDetection)
    monkeypatch.setattr(experiments, "CloudContinualLearner", DummyLearner)
    monkeypatch.setattr(
        experiments,
        "_build_split_runtime",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("No split candidate satisfies the fixed split constraints.")),
    )

    result = experiments._run_pair_experiment(
        base_config=_base_config(),
        video_path=Path("./video_data/road.mp4"),
        output_root=tmp_path,
        edge_model="tinynext_s",
        golden_model="rtdetr_x",
        max_samples=1,
        frame_stride=300,
        epochs=2,
        teacher_threshold=None,
        send_low_conf_features=True,
        refresh_weights=False,
    )

    assert result.success is False
    assert result.accepted_updated_weights is False
    assert result.before_proxy_map is None
    assert result.after_proxy_map is None
    assert result.split_config_id is None
    assert result.split_index is None
    assert result.payload_bytes is None
    assert result.sample_stats == {}
    assert "raw_fallback" not in result.sample_stats
    assert "No split candidate satisfies the fixed split constraints." in result.message
