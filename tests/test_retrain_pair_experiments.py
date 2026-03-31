from __future__ import annotations

import base64
import copy
import io
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from tools import run_retrain_pair_experiments as experiments


class _DummyDetectionModel(torch.nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.loaded_state = None

    def load_state_dict(self, state_dict, strict: bool = True):
        self.loaded_state = state_dict
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    def to(self, device):
        return self

    def eval(self):
        return self


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


def test_run_raw_retrain_fallback_reports_proxy_map_and_writes_frames(tmp_path, monkeypatch):
    frames = [
        np.full((8, 8, 3), 20, dtype=np.uint8),
        np.full((8, 8, 3), 40, dtype=np.uint8),
    ]
    frame_indices = [300, 600]
    before_model = _DummyDetectionModel("before")
    after_model = _DummyDetectionModel("after")

    payload = io.BytesIO()
    torch.save({"weight": torch.tensor([2.0])}, payload)

    class DummyLearner:
        device = torch.device("cpu")

        def __init__(self):
            self._loads = 0

        def _build_teacher_targets(self, frame):
            return {"boxes": [[1.0, 1.0, 4.0, 4.0]], "labels": [1]}

        def _load_edge_training_model(self, model_name=None):
            self._loads += 1
            return before_model if self._loads == 1 else after_model

        def get_ground_truth_and_retrain(self, **kwargs):
            return (
                True,
                base64.b64encode(payload.getvalue()).decode("utf-8"),
                "Retraining successful",
            )

    eval_results = [
        {
            "map": 0.1,
            "evaluated_samples": 2,
            "skipped_empty_gt": 0,
            "skipped_missing_frame": 0,
        },
        {
            "map": 0.4,
            "evaluated_samples": 2,
            "skipped_empty_gt": 0,
            "skipped_missing_frame": 0,
        },
    ]

    monkeypatch.setattr(
        experiments,
        "_evaluate_detection_proxy_map",
        lambda *args, **kwargs: copy.deepcopy(eval_results.pop(0)),
    )

    result = experiments._run_raw_retrain_fallback(
        learner=DummyLearner(),
        pair_root=tmp_path,
        frame_indices=frame_indices,
        frames=frames,
        edge_model="tinynext_s",
        epochs=2,
        failure_reason="No split candidate satisfies the fixed split constraints.",
    )

    success, accepted, before_map, after_map, absolute_delta, relative_delta_percent, sample_stats, message = result
    assert success is True
    assert accepted is True
    assert before_map == 0.1
    assert after_map == 0.4
    assert absolute_delta == pytest.approx(0.3)
    assert relative_delta_percent == pytest.approx(300.0)
    assert sample_stats["raw_fallback"] is True
    assert sample_stats["gt_annotation_count"] == 2
    assert "proxy_mAP@0.5 0.1000 -> 0.4000" in message
    assert (tmp_path / "raw_cache" / "frames" / "300.jpg").is_file()
    assert (tmp_path / "raw_cache" / "frames" / "600.jpg").is_file()
    assert set(after_model.loaded_state.keys()) == {"weight"}
    assert torch.equal(after_model.loaded_state["weight"], torch.tensor([2.0]))


def test_run_pair_experiment_falls_back_to_raw_retraining_when_split_plan_fails(tmp_path, monkeypatch):
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
    monkeypatch.setattr(
        experiments,
        "_run_raw_retrain_fallback",
        lambda **kwargs: (
            True,
            True,
            0.2,
            0.5,
            0.3,
            150.0,
            {"total_samples": 1, "raw_fallback": True},
            "Fell back to full-image retraining because fixed split was unavailable: No split candidate satisfies the fixed split constraints.; proxy_mAP@0.5 0.2000 -> 0.5000 (delta=+0.3000, evaluated=1, skipped_empty_gt=0, skipped_missing_frame=0)",
        ),
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

    assert result.success is True
    assert result.accepted_updated_weights is True
    assert result.before_proxy_map == 0.2
    assert result.after_proxy_map == 0.5
    assert result.split_config_id is None
    assert result.split_index is None
    assert result.payload_bytes is None
    assert result.sample_stats["raw_fallback"] is True
    assert "Fell back to full-image retraining" in result.message


def test_run_raw_retrain_fallback_rejects_dead_detector_updates(tmp_path, monkeypatch):
    frames = [np.full((8, 8, 3), 20, dtype=np.uint8)]
    frame_indices = [300]
    before_model = _DummyDetectionModel("before")
    after_model = _DummyDetectionModel("after")

    payload = io.BytesIO()
    torch.save({"weight": torch.tensor([3.0])}, payload)

    class DummyLearner:
        device = torch.device("cpu")

        def __init__(self):
            self._loads = 0

        def _build_teacher_targets(self, frame):
            return {"boxes": [[1.0, 1.0, 4.0, 4.0]], "labels": [1]}

        def _load_edge_training_model(self, model_name=None):
            self._loads += 1
            return before_model if self._loads == 1 else after_model

        def get_ground_truth_and_retrain(self, **kwargs):
            return (
                True,
                base64.b64encode(payload.getvalue()).decode("utf-8"),
                "Retraining successful",
            )

    eval_results = [
        {
            "map": 0.0,
            "evaluated_samples": 1,
            "skipped_empty_gt": 0,
            "skipped_missing_frame": 0,
            "nonempty_predictions": 1,
        },
        {
            "map": 0.0,
            "evaluated_samples": 1,
            "skipped_empty_gt": 0,
            "skipped_missing_frame": 0,
            "nonempty_predictions": 0,
        },
    ]

    monkeypatch.setattr(
        experiments,
        "_evaluate_detection_proxy_map",
        lambda *args, **kwargs: copy.deepcopy(eval_results.pop(0)),
    )

    result = experiments._run_raw_retrain_fallback(
        learner=DummyLearner(),
        pair_root=tmp_path,
        frame_indices=frame_indices,
        frames=frames,
        edge_model="tinynext_s",
        epochs=2,
        failure_reason="No split candidate satisfies the fixed split constraints.",
    )

    success, accepted, before_map, after_map, absolute_delta, relative_delta_percent, sample_stats, message = result
    assert success is True
    assert accepted is False
    assert before_map == 0.0
    assert after_map == 0.0
    assert absolute_delta == pytest.approx(0.0)
    assert relative_delta_percent is None
    assert sample_stats["raw_fallback"] is True
    assert "Experiment rejected the updated weights because updated weights produced no detections" in message
