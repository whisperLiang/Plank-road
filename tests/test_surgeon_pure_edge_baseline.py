from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from baselines.runtime import BaselineEdgeAdapter


def _config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        source=SimpleNamespace(video_path="video.mp4"),
        lightweight="toy-detector",
        baseline=SimpleNamespace(
            results_root=str(tmp_path / "results"),
            pure_edge_local_updating=SimpleNamespace(
                label_source="pseudo_label",
                local_metrics=True,
                upload_metrics_to_cloud=False,
                upload_frames_to_cloud=False,
                use_cloud_teacher=False,
                training_strategy="surgeon_tta",
                quality_mode="output_only_when_no_boundary",
                tta_steps=1,
                trigger_low_quality_samples=2,
                max_local_buffer_samples=8,
                trainable_scope="norm_affine",
                consistency_weight=0.0,
                entropy_margin_ratio=1.0,
            ),
            accuracy_trigger_cloud_retraining=SimpleNamespace(
                training_strategy="freeze",
                trainable_param_ratio=0.3,
                training_failure_backoff_sec=30.0,
            ),
            training=SimpleNamespace(
                batch_size=2,
                learning_rate=1.0e-2,
                weight_decay=0.0,
                optimizer_name="adam",
                min_training_samples=1,
                training_window_size=8,
            ),
            edge=SimpleNamespace(split_runtime_policy="disabled"),
        ),
        sample_quality=SimpleNamespace(
            enabled=True,
            output_entropy=SimpleNamespace(
                window_size=8,
                percentile=25.0,
                warmup_samples=0,
                min_detection_confidence=0.0,
            ),
            boundary_feature_entropy=SimpleNamespace(
                max_elements=16,
                ema_decay=0.95,
                deviation_threshold=1.5,
                min_std=1.0e-4,
                warmup_samples=0,
            ),
            eps=1.0e-8,
            persist_debug_stats=False,
        ),
        window_drift=SimpleNamespace(
            window_size=4,
            min_window_size=1,
            low_quality_rate_threshold=0.5,
            persistence_windows=1,
        ),
        das=SimpleNamespace(
            enabled=False,
            bn_only=True,
            probe_samples=2,
            strategy="tgi",
            use_spectral_entropy=False,
        ),
    )


class ToyTTAModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(3)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.head = torch.nn.Linear(3, 3)
        self.skip_logits = False
        self.fail_forward = False

    def forward_tta_outputs(self, images, *, augment: bool = False):
        del augment
        if self.fail_forward:
            raise RuntimeError("forced tta failure")
        batch = torch.stack([image.float() for image in images])
        features = self.pool(self.bn(batch)).flatten(1)
        logits = self.head(features)
        if self.skip_logits:
            return {"scores": logits.detach().softmax(dim=-1).max(dim=-1).values}
        return {"logits": logits}


class FakeDetector:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.model_lock = threading.Lock()

    def _prepare_image_tensor(self, frame):
        tensor = torch.from_numpy(np.ascontiguousarray(frame[..., ::-1].copy()))
        return tensor.permute(2, 0, 1).float().div_(255.0)


class FakeEdge:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model_version = "0"
        self.small_object_detection = FakeDetector(model)
        self.apply_model_update_calls = 0

    def apply_model_update(self, *args, **kwargs) -> None:
        del args, kwargs
        self.apply_model_update_calls += 1
        raise AssertionError("pure edge local TTA must not call apply_model_update")


class FakeTask:
    def __init__(self, frame_id: int) -> None:
        self.result_source = "inference"
        self.timing_ms = {"inference": 1.0}
        self.inference_artifacts = {
            "boxes": [],
            "labels": [],
            "scores": [],
            "confidence": 0.0,
            "entropy": 0.9,
            "model_version": "0",
            "result_source": "inference",
            "frame_id": frame_id,
        }


def _adapter(tmp_path: Path) -> BaselineEdgeAdapter:
    return BaselineEdgeAdapter(
        config=_config(tmp_path),
        baseline_method="pure_edge_local_updating",
        run_id="pure-surgeon-test",
        edge_id=1,
        transport=None,
    )


def _frame(value: int) -> np.ndarray:
    return np.full((8, 8, 3), value, dtype=np.uint8)


def _sample(adapter: BaselineEdgeAdapter, frame_id: int) -> None:
    adapter.on_sampled_inference_result(
        frame=_frame(frame_id),
        frame_index=frame_id,
        task=FakeTask(frame_id),
        detection_boxes=[],
        detection_class=[],
        detection_score=[],
        latency_ms=1.0,
    )


def _metrics(adapter: BaselineEdgeAdapter) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in Path(adapter.metrics_path).read_text(encoding="utf-8").splitlines()
    ]


class RFDETRLikeWrapper(torch.nn.Module):
    def __init__(self, inner: ToyTTAModel) -> None:
        super().__init__()
        self.rfdetr = SimpleNamespace(model=SimpleNamespace(model=inner))

    def forward_tta_outputs(self, images, *, augment: bool = False):
        return self.rfdetr.model.model.forward_tta_outputs(images, augment=augment)


def test_pure_edge_surgeon_keeps_transport_none_and_records_frame_decision(tmp_path) -> None:
    model = ToyTTAModel()
    adapter = _adapter(tmp_path)
    try:
        adapter.before_video_start(FakeEdge(model))
        _sample(adapter, 1)

        rows = _metrics(adapter)
        decision = next(row for row in rows if row["event"] == "frame_decision")
        assert adapter.transport is None
        assert adapter._worker is None
        assert adapter._training_state is None
        assert decision["upload_frame"] is False
        assert decision["upload_mode"] == "none"
        assert decision["training_strategy"] == "surgeon_tta"
    finally:
        adapter.close()


def test_tta_selects_inner_trainable_module_for_detector_wrappers(tmp_path) -> None:
    inner = ToyTTAModel()
    inner.eval()
    wrapper = RFDETRLikeWrapper(inner)
    edge = FakeEdge(wrapper)
    adapter = _adapter(tmp_path)
    try:
        adapter.before_video_start(edge)
        _sample(adapter, 1)
        _sample(adapter, 2)
        assert adapter._surgeon_tta is not None
        assert adapter._surgeon_tta.wait_for_idle(timeout=5.0)

        rows = _metrics(adapter)
        done = next(row for row in rows if row["event"] == "surgeon_tta_done")
        assert done["trainable_param_count"] > 0
        assert done["model_version_after"] == "surgeon_1"
        assert edge.model_version == "surgeon_1"
        assert inner.training is False
    finally:
        adapter.close()


def test_low_quality_samples_trigger_local_tta_and_update_model_version(tmp_path) -> None:
    model = ToyTTAModel()
    model.eval()
    edge = FakeEdge(model)
    adapter = _adapter(tmp_path)
    try:
        adapter.before_video_start(edge)
        _sample(adapter, 1)
        _sample(adapter, 2)
        assert adapter._surgeon_tta is not None
        assert adapter._surgeon_tta.wait_for_idle(timeout=5.0)

        rows = _metrics(adapter)
        events = [row["event"] for row in rows]
        done = next(row for row in rows if row["event"] == "surgeon_tta_done")
        assert "surgeon_tta_triggered" in events
        assert "surgeon_tta_started" in events
        assert "local_model_update_applied" in events
        assert done["model_version_before"] == "0"
        assert done["model_version_after"] == "surgeon_1"
        assert done["trainable_param_count"] > 0
        assert edge.model_version == "surgeon_1"
        assert edge.apply_model_update_calls == 0
        assert model.training is False
    finally:
        adapter.close()


def test_logits_unavailable_skips_without_cloud_update_and_restores_mode(tmp_path) -> None:
    model = ToyTTAModel()
    model.skip_logits = True
    model.eval()
    edge = FakeEdge(model)
    adapter = _adapter(tmp_path)
    try:
        adapter.before_video_start(edge)
        _sample(adapter, 1)
        _sample(adapter, 2)
        assert adapter._surgeon_tta is not None
        assert adapter._surgeon_tta.wait_for_idle(timeout=5.0)

        _sample(adapter, 3)
        rows = _metrics(adapter)
        skipped = next(row for row in rows if row["event"] == "surgeon_tta_skipped")
        assert skipped["reason"] == "logits_unavailable"
        assert edge.model_version == "0"
        assert edge.apply_model_update_calls == 0
        assert model.training is False
        assert sum(1 for row in rows if row["event"] == "frame_decision") >= 3
    finally:
        adapter.close()


def test_missing_model_lock_skips_without_private_lock(tmp_path) -> None:
    model = ToyTTAModel()
    model.eval()
    edge = FakeEdge(model)
    delattr(edge.small_object_detection, "model_lock")
    adapter = _adapter(tmp_path)
    try:
        adapter.before_video_start(edge)
        _sample(adapter, 1)
        _sample(adapter, 2)
        assert adapter._surgeon_tta is not None
        assert adapter._surgeon_tta.wait_for_idle(timeout=5.0)

        _sample(adapter, 3)
        rows = _metrics(adapter)
        skipped = next(row for row in rows if row["event"] == "surgeon_tta_skipped")
        assert skipped["reason"] == "model_lock_unavailable"
        assert edge.model_version == "0"
        assert model.training is False
        assert sum(1 for row in rows if row["event"] == "frame_decision") >= 3
    finally:
        adapter.close()


def test_failed_tta_restores_mode_and_later_inference_continues(tmp_path) -> None:
    model = ToyTTAModel()
    model.fail_forward = True
    model.eval()
    edge = FakeEdge(model)
    adapter = _adapter(tmp_path)
    try:
        adapter.before_video_start(edge)
        _sample(adapter, 1)
        _sample(adapter, 2)
        assert adapter._surgeon_tta is not None
        assert adapter._surgeon_tta.wait_for_idle(timeout=5.0)

        _sample(adapter, 3)
        rows = _metrics(adapter)
        assert any(row["event"] == "surgeon_tta_failed" for row in rows)
        assert model.training is False
        assert sum(1 for row in rows if row["event"] == "frame_decision") >= 3
    finally:
        adapter.close()
